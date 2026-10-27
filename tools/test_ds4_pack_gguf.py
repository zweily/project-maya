"""tools/test_ds4_pack_gguf.py - the pack's DSpark geometry fill (tools/ds4/pack_gguf.py's dspark_fill): a GGUF quant
of the base model carries the dspark.* metadata or not, and the drafter's own record (a checkpoint config.json kept
by drafter_pack.py) fills exactly the holes - never overwriting what the GGUF states, 0 counting as a hole because 0
is what turns the drafter off in the engine without a word.

    python -m unittest tools.test_ds4_pack_gguf -v
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools" / "ds4")]
import drafter_pack  # noqa: E402
import pack_gguf  # noqa: E402

CFG = {"dspark_block_size": 5, "dspark_noise_token_id": 128799, "dspark_markov_rank": 256,
       "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3, "dspark_target_layer_ids": [37, 38, 39]}


class Fill(unittest.TestCase):
    def test_a_metadata_free_gguf_is_filled_whole(self):
        kv = {"general.architecture": "deepseek41"}
        added = pack_gguf.dspark_fill(kv, "deepseek41", CFG)
        self.assertEqual(len(added), 6)
        self.assertEqual(kv["deepseek41.dspark.block_size"], 5)
        self.assertEqual(kv["deepseek41.dspark.target_layers"], [37, 38, 39])

    def test_what_the_gguf_states_stands_and_holes_fill(self):
        kv = {"deepseek41.dspark.block_size": 7,          # its own: wins over the checkpoint's 5
              "deepseek41.dspark.markov_rank": 0}         # a stated 0 is a hole: it silences the drafter
        added = pack_gguf.dspark_fill(kv, "deepseek41", CFG)
        self.assertEqual(kv["deepseek41.dspark.block_size"], 7)
        self.assertEqual(kv["deepseek41.dspark.markov_rank"], 256)
        self.assertNotIn("deepseek41.dspark.block_size", added)
        self.assertIn("deepseek41.dspark.markov_rank", added)
        self.assertEqual(len(added), 5)                   # five holes, block_size had none

    def test_a_drafter_without_a_record_changes_nothing(self):
        kv = {"deepseek41.dspark.block_size": 5}
        self.assertEqual(pack_gguf.dspark_fill(kv, "deepseek41", {}), [])
        self.assertEqual(kv, {"deepseek41.dspark.block_size": 5})


class DrafterRecord(unittest.TestCase):
    """drafter_pack.py's side of the chain: where config.json is looked for, and what lands in dspark.json."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.base = self.root / "DeepSeek-V4.1-Flash-1017"        # the checkpoint dir the drafter sits beside
        self.d = self.base / "dspark"
        self.d.mkdir(parents=True)
        self.addCleanup(self.tmp.cleanup)

    def _write(self, p, cfg):
        import json
        p.write_text(json.dumps(cfg))

    def test_the_nested_official_shape_is_walked(self):
        self._write(self.base / "config.json", {"text_config": {"dspark_block_size": 5, "hidden_size": 7168,
                                                                 "sub": {"dspark_target_layer_ids": [37, 38, 39]}}})
        cfg = drafter_pack.drafter_config(drafter_pack.find_config(self.d))
        self.assertEqual(cfg, {"config": {"dspark_block_size": 5, "dspark_target_layer_ids": [37, 38, 39]}})

    def test_search_order_beside_then_inside_then_nowhere(self):
        self.assertIsNone(drafter_pack.find_config(self.d))                     # nothing around
        self._write(self.d / "config.json", {"dspark_block_size": 1})
        self.assertEqual(drafter_pack.find_config(self.d), self.d / "config.json")
        self._write(self.base / "config.json", {"dspark_block_size": 2})
        self.assertEqual(drafter_pack.find_config(self.d), self.d / "config.json")   # the folder's own wins
        self.assertEqual(drafter_pack.find_config(self.base / "dspark2"), self.base / "config.json")
        self.assertEqual(drafter_pack.drafter_config(self.base / "config.json"), {"config": {"dspark_block_size": 2}})

    def test_a_config_without_dspark_keys_records_nothing(self):
        self._write(self.d / "config.json", {"text_config": {"hidden_size": 7168}})
        self.assertEqual(drafter_pack.drafter_config(drafter_pack.find_config(self.d)), {})

    def test_an_explicit_config_that_is_not_there_is_an_error(self):
        self.assertRaises(SystemExit, drafter_pack.find_config, self.d, self.d / "nowhere.json")

    def test_the_setup_downloads_the_config_it_looks_for(self):
        """the fresh-setup half of the chain: download_gguf.json's drafter carries the checkpoint's config.json into
        the drafter's folder - the file find_config then finds without being told - with its size and hash."""
        man = json.loads((ROOT / "tools" / "ds4" / "download_gguf.json").read_text())["drafter"]
        size, sha = man["files"]["config.json"]
        self.assertGreater(size, 0)
        self.assertRegex(sha, r"^[0-9a-f]{64}$")


class RealCheckpointConfig(unittest.TestCase):
    """the shape DeepSeek actually ships (a text_config nest, the six dspark_* keys) against the extractor."""

    def test_the_six_keys_the_engine_wants_come_out_of_the_nested_file(self):
        import tempfile
        cfg = {"architectures": ["X"], "text_config": {"dspark_block_size": 5, "dspark_noise_token_id": 128799,
               "dspark_markov_rank": 256, "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3,
               "dspark_target_layer_ids": [37, 38, 39], "hidden_size": 7168}}
        with tempfile.TemporaryDirectory() as t:
            p = pathlib.Path(t) / "config.json"
            p.write_text(json.dumps(cfg))
            got = drafter_pack.drafter_config(p)
        kv = {"general.architecture": "deepseek41", "deepseek41.dspark.block_size": 0}
        self.assertEqual(len(pack_gguf.dspark_fill(kv, "deepseek41", got["config"])), 6)
        self.assertEqual(kv["deepseek41.dspark.block_size"], 5)      # the stated 0 was a hole and got filled


class MergeMode(unittest.TestCase):
    """an install predating the geometry: the folder holds a dspark.json without config and no shards (they were
    converted and deleted by the old setup); drafter_pack.py --dir on it merges, and a second run is a no-op."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name) / "dspark"
        self.d.mkdir()
        (self.d / "dspark.json").write_text(json.dumps({"from": ["model-00044-of-00048.safetensors"], "tensors": []}))
        (self.d / "config.json").write_text(json.dumps({"text_config": CFG}))
        self.addCleanup(self.tmp.cleanup)

    def _main(self):
        import unittest.mock
        with unittest.mock.patch.object(sys, "argv", ["drafter_pack.py", "--dir", str(self.d)]):
            return drafter_pack.main()

    def test_merge_then_no_op(self):
        self.assertEqual(self._main(), 0)
        dj = json.loads((self.d / "dspark.json").read_text())
        self.assertEqual(dj["config"]["dspark_block_size"], 5)
        self.assertEqual(dj["tensors"], [])                       # (nothing else was touched)
        self.assertEqual(list(dj)[:2], ["from", "config"])        # (a merge reads like a fresh conversion wrote it)
        (self.d / "config.json").unlink()                          # the merge kept the record; the file may go
        self.assertEqual(self._main(), 0)                          # (config now lives in dspark.json's own folder...)
        self.assertEqual(json.loads((self.d / "dspark.json").read_text())["config"]["dspark_block_size"], 5)

    def test_a_half_record_is_finished_not_declared_done(self):
        (self.d / "dspark.json").write_text(json.dumps(
            {"from": ["x"], "config": {"dspark_block_size": 7}, "tensors": []}))
        self.assertEqual(self._main(), 0)
        cfg = json.loads((self.d / "dspark.json").read_text())["config"]
        self.assertEqual(cfg["dspark_block_size"], 7)             # (what the record states stands)
        self.assertEqual(cfg["dspark_noise_token_id"], 128799)    # (what it lacked came from the checkpoint)

    def test_no_shards_and_no_config_is_an_error(self):
        (self.d / "config.json").unlink()
        self.assertRaises(SystemExit, self._main)

    def test_an_explicit_config_is_honoured_where_the_defaults_would_not_find_one(self):
        self.d.parent.joinpath("elsewhere.json").write_text(json.dumps({"dspark_block_size": 9}))
        self.assertEqual(drafter_pack.find_config(self.d, self.d.parent / "elsewhere.json"),
                         self.d.parent / "elsewhere.json")
        self.assertEqual(drafter_pack.drafter_config(self.d.parent / "elsewhere.json"),
                         {"config": {"dspark_block_size": 9}})


if __name__ == "__main__":
    unittest.main()
