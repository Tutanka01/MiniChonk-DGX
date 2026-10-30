import importlib.util
import json
import os
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

from minichonk import gguf, trim

ROOT = pathlib.Path(__file__).resolve().parents[1]
N_EXPERT = 8
KEPT = {1: [0, 2, 5, 7], 2: [1, 3, 4, 6]}


def f32(values):
    return struct.pack(f"<{len(values)}f", *values)


def expert_tensor_bytes(dim, layer, role):
    """Every value encodes (layer, role, expert) so a wrong slab is detectable."""
    out = bytearray()
    for expert in range(N_EXPERT):
        out += f32([layer * 100 + role * 10 + expert] * (dim * dim))
    return bytes(out)


def build_shard(path, tensors, kvs, alignment=32):
    infos, offset = [], 0
    for name, dims, data in tensors:
        infos.append(gguf.TensorInfo(name, list(dims), 0, offset))
        offset = gguf.align_up(offset + len(data), alignment)
    header = gguf.Header(3, kvs, infos, alignment, 0)
    encoded = gguf.encode_header(header)
    with open(path, "wb") as f:
        f.write(encoded)
        written = 0
        for (name, dims, data), info in zip(tensors, infos):
            f.write(b"\0" * (info.offset - written))
            f.write(data)
            written = info.offset + len(data)


def uint16_kv(key, value):
    return gguf.KV(key, gguf.KV_UINT16, gguf.encode_scalar(gguf.KV_UINT16, value), value)


def uint32_kv(key, value):
    return gguf.KV(key, gguf.KV_UINT32, gguf.encode_scalar(gguf.KV_UINT32, value), value)


def model_tensors():
    dim = 4
    tensors = [("blk.0.ffn_up.weight", [dim, dim], f32([0.5] * dim * dim))]
    for layer in (1, 2):
        # attn_v_b deliberately has N_EXPERT as its last dim: it must survive untouched.
        tensors.append((f"blk.{layer}.attn_v_b.weight", [dim, N_EXPERT], f32([9.0 + layer] * dim * N_EXPERT)))
        for role, name in enumerate(("gate", "down", "up")):
            tensors.append((f"blk.{layer}.ffn_{name}_exps.weight", [dim, dim, N_EXPERT],
                            expert_tensor_bytes(dim, layer, role)))
        router = bytearray()
        for expert in range(N_EXPERT):
            router += f32([layer * 1000 + expert] * dim)
        tensors.append((f"blk.{layer}.ffn_gate_inp.weight", [dim, N_EXPERT], bytes(router)))
    return dim, tensors


def read_tensor(path, name):
    header = gguf.parse_file(path)
    t = next(x for x in header.tensors if x.name == name)
    with open(path, "rb") as f:
        f.seek(header.data_start + t.offset)
        data = f.read(t.nbytes())
    return t, struct.unpack(f"<{len(data) // 4}f", data)


class TrimTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.dir.name)
        self.src = self.root / "src"
        self.src.mkdir()
        self.dst = self.root / "dst"
        self.dim, tensors = model_tensors()
        kvs = [uint32_kv("general.alignment", 32), uint32_kv("mistral-large4.expert_count", N_EXPERT),
               uint32_kv("mistral-large4.block_count", 3), uint16_kv("split.count", 2)]
        build_shard(self.src / "m-00001-of-00002.gguf", tensors[:6], kvs)
        build_shard(self.src / "m-00002-of-00002.gguf", tensors[6:], [uint32_kv("split.no", 1)])
        self.mask = self.root / "mask.json"
        self.mask.write_text(json.dumps({"layers": {str(k): {"kept": v} for k, v in KEPT.items()}}))

    def tearDown(self):
        self.dir.cleanup()

    def run_trim(self, *extra):
        trim.main(["--src", str(self.src), "--dst", str(self.dst), "--mask", str(self.mask),
                   "--n-expert", str(N_EXPERT), *extra])

    def tensor_anywhere(self, name):
        for path in sorted(self.dst.glob("*.gguf")):
            if any(t.name == name for t in gguf.parse_file(path).tensors):
                return read_tensor(path, name)
        self.fail(f"{name} missing")

    def test_expert_slabs_keep_mask_order(self):
        self.run_trim("--verify")
        for layer, kept in KEPT.items():
            for role, name in enumerate(("gate", "down", "up")):
                t, values = self.tensor_anywhere(f"blk.{layer}.ffn_{name}_exps.weight")
                self.assertEqual(t.dims, [self.dim, self.dim, len(kept)])
                per_slab = self.dim * self.dim
                got = [values[i * per_slab] for i in range(len(kept))]
                self.assertEqual(got, [layer * 100 + role * 10 + e for e in kept])
                self.assertEqual(len(values), per_slab * len(kept))

    def test_router_keeps_same_experts(self):
        self.run_trim()
        for layer, kept in KEPT.items():
            t, values = self.tensor_anywhere(f"blk.{layer}.ffn_gate_inp.weight")
            self.assertEqual(t.dims, [self.dim, len(kept)])
            self.assertEqual([values[i * self.dim] for i in range(len(kept))], [layer * 1000 + e for e in kept])

    def test_coincidental_dim_is_untouched(self):
        self.run_trim()
        t, values = self.tensor_anywhere("blk.1.attn_v_b.weight")
        self.assertEqual(t.dims, [self.dim, N_EXPERT])
        self.assertEqual(set(values), {10.0})

    def test_metadata(self):
        self.run_trim()
        first = gguf.parse_file(self.dst / "m-00001-of-00002.gguf")
        self.assertEqual(first.kv("mistral-large4.expert_count").value, 4)
        self.assertEqual(first.kv("mistral-large4.block_count").value, 3)
        self.assertEqual(gguf.parse_file(self.dst / "m-00002-of-00002.gguf").kv("split.no").value, 1)

    def test_outputs_pass_layout_check(self):
        self.run_trim()
        for path in self.dst.glob("*.gguf"):
            gguf.check_layout(gguf.parse_file(path), os.path.getsize(path))

    def test_rerun_skips_finished_shards(self):
        self.run_trim()
        before = {p.name: p.stat().st_mtime_ns for p in self.dst.glob("*.gguf")}
        self.run_trim()
        self.assertEqual(before, {p.name: p.stat().st_mtime_ns for p in self.dst.glob("*.gguf")})

    def test_verify_detects_corruption(self):
        self.run_trim()
        path = self.dst / "m-00001-of-00002.gguf"
        header = gguf.parse_file(path)
        t = next(x for x in header.tensors if x.name == "blk.1.ffn_up_exps.weight")
        with open(path, "r+b") as f:
            f.seek(header.data_start + t.offset)
            f.write(b"\xff\xff\xff\xff")
        with self.assertRaises(gguf.GGUFError):
            trim.verify_shard(self.src / path.name, path, KEPT, N_EXPERT)

    def test_rejects_layer_missing_from_mask(self):
        self.mask.write_text(json.dumps({"layers": {"1": {"kept": KEPT[1]}}}))
        with self.assertRaises(gguf.GGUFError):
            self.run_trim()

    def test_rejects_unsorted_mask(self):
        self.mask.write_text(json.dumps({"layers": {"1": {"kept": [2, 1]}, "2": {"kept": [1, 2]}}}))
        with self.assertRaises(gguf.GGUFError):
            self.run_trim()

    def test_rejects_wrong_expert_count(self):
        with self.assertRaises(gguf.GGUFError):
            trim.main(["--src", str(self.src), "--dst", str(self.dst), "--mask", str(self.mask), "--n-expert", "16"])


class HardeningTest(TrimTest):
    def test_rejects_incomplete_source(self):
        (self.src / "m-00002-of-00002.gguf").unlink()
        with self.assertRaisesRegex(gguf.GGUFError, "announces 2"):
            self.run_trim()

    def test_rejects_other_mask_on_same_output(self):
        self.run_trim()
        self.mask.write_text(json.dumps({"layers": {"1": {"kept": [0, 1, 2, 3]}, "2": {"kept": [0, 1, 2, 3]}}}))
        with self.assertRaisesRegex(gguf.GGUFError, "another mask"):
            self.run_trim()

    def test_rebuilds_truncated_shard(self):
        self.run_trim()
        path = self.dst / "m-00002-of-00002.gguf"
        expected = path.stat().st_size
        with open(path, "r+b") as f:
            f.truncate(expected - 10)
        self.run_trim("--verify")
        self.assertEqual(path.stat().st_size, expected)

    def test_rejects_when_disk_is_too_small(self):
        with mock.patch("shutil.disk_usage", return_value=mock.Mock(free=10)):
            with self.assertRaisesRegex(gguf.GGUFError, "GB free"):
                self.run_trim()


class GGUFTest(unittest.TestCase):
    def test_header_round_trip_keeps_metadata(self):
        kvs = [uint32_kv("a", 7), gguf.KV("b", gguf.KV_STRING, gguf.encode_string("é"), "é")]
        header = gguf.Header(3, kvs, [gguf.TensorInfo("t", [4, 2], 0, 0)], 32, 0)
        encoded = gguf.encode_header(header)
        parsed = gguf.parse_header(lambda offset, size: encoded[offset:offset + size])
        self.assertEqual(parsed.kv("a").value, 7)
        self.assertEqual(parsed.kv("b").value, "é")
        self.assertEqual(parsed.tensors[0].dims, [4, 2])
        self.assertEqual(parsed.data_start, len(encoded))

    def test_rejects_other_files(self):
        with self.assertRaises(gguf.GGUFError):
            gguf.parse_header(lambda offset, size: b"NOPE\0\0\0\0"[offset:offset + size])

    def test_check_layout_detects_wrong_offset(self):
        header = gguf.Header(3, [], [gguf.TensorInfo("a", [4], 0, 0), gguf.TensorInfo("b", [4], 0, 99)], 32, 64)
        with self.assertRaises(gguf.GGUFError):
            gguf.check_layout(header)

    def test_quantized_size_uses_blocks(self):
        # UD-IQ1_S experts: 6144 x 6144 x 128 at 50 bytes per 256 elements.
        t = gguf.TensorInfo("t", [6144, 6144, 128], 19, 0)
        self.assertEqual(t.slab_bytes(), 6144 // 256 * 50 * 6144)


class ResultsTest(unittest.TestCase):
    def test_raw_files_give_published_scores(self):
        spec = importlib.util.spec_from_file_location("summarize_results", ROOT / "scripts" / "summarize_results.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        best = module.collect()["aimer112"]
        self.assertAlmostEqual(best["perplexity"], 5.3065, places=3)
        self.assertEqual(best["mmlu"], (200, 47.0))
        self.assertEqual(best["arc"], (200, 63.0))


class ClusterScriptTest(unittest.TestCase):
    def test_never_kills_by_command_line(self):
        # Matching on the full command line also hits the tmux server that started the process.
        script = (ROOT / "scripts" / "cluster.sh").read_text()
        self.assertNotIn("pkill -f", script)

    def test_uses_private_tmux_socket(self):
        script = (ROOT / "scripts" / "cluster.sh").read_text()
        self.assertNotRegex(script, r'(?<!TMUX_CMD=)"tmux ')


class MaskTest(unittest.TestCase):
    def test_scores_reproduce_published_masks(self):
        spec = importlib.util.spec_from_file_location("make_aimer_mask", ROOT / "scripts" / "make_aimer_mask.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scores = json.loads((ROOT / "masks" / "aimer_scores.json").read_text())["scores"]
        for keep, name in ((96, "letrim25"), (64, "letrim50")):
            published = json.loads((ROOT / "masks" / f"{name}.json").read_text())["layers"]
            built = module.build(scores, keep)
            self.assertEqual({k: v["kept"] for k, v in built.items()}, {k: v["kept"] for k, v in published.items()})

    def test_shipped_masks_are_valid(self):
        for path in (ROOT / "masks").glob("*.json"):
            if path.name in ("aimer_scores.json", "expert_usage.json"):
                continue
            layers = trim.load_mask(path, 128)
            self.assertEqual(sorted(layers), list(range(1, 72)), path.name)


if __name__ == "__main__":
    unittest.main()
