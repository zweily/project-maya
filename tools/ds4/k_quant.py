"""tools/ds4/k_quant.py - ggml K-quant/IQ4_NL dequantization and Q8_0 quantization in numpy, for pack_gguf.py's
--convert: a GGUF whose dense matrices the engine reads only as BF16/Q8_0 gets them rewritten in place of a kernel
change; the added Q8_0 grid error sits two quantization steps under the source format's own.

dequant(f32): q4_K, q5_K, q6_K, iq4_nl, q8_0.  quantize: q8_0 (ggml's quantize_row_q8_0_ref: d = amax / 127 as F16,
values round half away from zero).  The block layouts and the value walks follow third_party/llama.cpp's ggml-common.h
and the scalar dequantize_row_* of ggml-quants.c; the float order of operations is left unfused - this is a pack pass,
bit-exactness belongs to the kernels.
"""
from __future__ import annotations

import numpy as np

QK_K, QK8_0 = 256, 32
BLK = {"Q4_K": 144, "Q5_K": 176, "Q6_K": 210, "IQ4_NL": 18, "Q8_0": 34}
VALS = {"Q4_K": QK_K, "Q5_K": QK_K, "Q6_K": QK_K, "IQ4_NL": 32, "Q8_0": QK8_0}

_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)


def _f16(b: np.ndarray) -> np.ndarray:
    return np.frombuffer(np.ascontiguousarray(b).tobytes(), np.float16).astype(np.float32)


def _dm(b: np.ndarray):
    """d and dmin: the F16 pair at the block's start (ggml's dm union)."""
    return _f16(b[:, 0:2]), _f16(b[:, 2:4])


def _scales_min(b: np.ndarray):
    """q4_K/q5_K's 12 scale/min bytes as 8 (scale, min) pairs - ggml's get_scale_min_k4."""
    s = b[:, 4:16].astype(np.uint32)
    sc = np.empty((len(s), 8), np.uint32)
    mn = np.empty_like(sc)
    sc[:, :4] = s[:, :4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0xF) | ((s[:, :4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def dequant(dtype: str, raw: bytes | np.ndarray, n: int) -> np.ndarray:
    """The n float32 values of a block-quantized tensor of this ggml dtype."""
    b = np.frombuffer(raw, np.uint8)
    nb, qk = n // VALS[dtype], VALS[dtype]
    assert len(b) == nb * BLK[dtype], f"{len(b)} bytes for {n} {dtype} values"
    if nb == 0:
        return np.zeros(n, np.float32)
    b = b.reshape(nb, BLK[dtype])

    if dtype == "Q8_0":
        d = _f16(b[:, 0:2])
        return (d[:, None] * b[:, 2:34].astype(np.int8)).reshape(n)

    if dtype == "IQ4_NL":
        d = _f16(b[:, 0:2])
        q = b[:, 2:18]
        lo = _IQ4NL[q & 0xF]                       # values j: the low nibbles of qs[j], j < 16
        hi = _IQ4NL[q >> 4]                        # values j + 16: their high nibbles
        return (np.concatenate([lo, hi], 1) * np.repeat(d[:, None], 32, 1)).reshape(n)

    d, dmin = _dm(b)
    sc, mn = _scales_min(b)
    d1 = d[:, None] * sc                           # the eight 32-value sub-blocks' scales and mins
    m1 = dmin[:, None] * mn

    if dtype in ("Q4_K", "Q5_K"):
        # ggml's walk: 64 values a step reads 32 bytes: values [0,32) the low nibbles (sub-block 2g), [32,64) the
        # high (2g + 1); q5_K adds its 5th bit: byte l of qh, bit 2g for the low half and bit 2g + 1 for the high.
        qs = b[:, BLK[dtype] - 128:]
        nib_lo = (qs & 0xF).astype(np.float32).reshape(nb, 4, 32)
        nib_hi = (qs >> 4).astype(np.float32).reshape(nb, 4, 32)
        if dtype == "Q5_K":
            qh = b[:, 16:48].astype(np.uint32)
            for g in range(4):
                nib_lo[:, g] += ((qh >> (2 * g) & 1) * 16).astype(np.float32)
                nib_hi[:, g] += ((qh >> (2 * g + 1) & 1) * 16).astype(np.float32)
        out = np.empty((nb, 4, 2, 32), np.float32)
        out[:, :, 0] = d1[:, 0::2, None] * nib_lo - m1[:, 0::2, None]
        out[:, :, 1] = d1[:, 1::2, None] * nib_hi - m1[:, 1::2, None]
        return out.reshape(n)

    if dtype == "Q6_K":
        d6 = _f16(b[:, 208:210])
        ql, qh, sc6 = b[:, 0:128], b[:, 128:192], b[:, 192:208].astype(np.int8).astype(np.float32)
        vals = np.empty((nb, 2, 4, 32), np.float32)   # [block][half h][quarter][l]
        for h in range(2):
            qlh, qhh = ql[:, h * 64:h * 64 + 64], qh[:, h * 32:h * 32 + 32]
            for quarter in range(4):
                src = qlh[:, np.arange(32) + 32 * (quarter & 1)]
                nib = src >> 4 if quarter & 2 else src & 0xF
                q = (nib.astype(np.int32) | (((qhh[:, np.arange(32)] >> (2 * quarter)).astype(np.int32)) & 3) << 4) - 32
                sch = sc6[:, h * 8 + 2 * quarter + (np.arange(32) >> 4)]
                vals[:, h, quarter] = d6[:, None] * sch * q.astype(np.float32)
        # ggml's stores land y[l], y[l+32], y[l+64], y[l+96] for a quarter of 32 l's: quarter q fills q*32..q*32+31
        return vals.reshape(n)

    raise ValueError(f"k_quant: no {dtype} dequantization")


def quant_q8_0(w: np.ndarray) -> bytes:
    """ggml's quantize_row_q8_0_ref: d = amax / 127 in F16, values round half away from zero."""
    v = np.ascontiguousarray(w, np.float32).reshape(-1, QK8_0)
    amax = np.abs(v).max(1)
    d = (amax / 127.0).astype(np.float32)
    idd = np.where(d > 0, 1.0 / np.where(d == 0, 1.0, d), 0.0)
    x = v * idd[:, None]
    q = (np.sign(x) * np.floor(np.abs(x) + 0.5)).clip(-127, 127).astype(np.int8)
    raw = np.empty((len(v), 34), np.uint8)
    raw[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    raw[:, 2:34] = q.view(np.uint8).reshape(len(v), 32)
    return raw.tobytes()


def _self_test() -> None:
    rng = np.random.default_rng(7)
    w = (rng.standard_normal(32 * 100) * 0.02).astype(np.float32)
    back = dequant("Q8_0", quant_q8_0(w), w.size)
    assert float(np.abs(back - w).max() / np.abs(w).max()) < 0.008, "Q8_0 round trip off its grid"
    for dt in ("Q4_K", "Q5_K", "Q6_K", "IQ4_NL"):
        blk = rng.integers(0, 256, size=(400, BLK[dt]), dtype=np.uint8)
        blk[:, 0:2] &= 0x7B                                  # keep the F16 scales finite
        v = dequant(dt, blk.tobytes(), 400 * VALS[dt])
        assert v.size == 400 * VALS[dt] and v.dtype == np.float32
    print("k_quant: self-test ok")


if __name__ == "__main__":
    _self_test()
