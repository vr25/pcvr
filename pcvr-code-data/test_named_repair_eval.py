import json
import tempfile
import unittest
from pathlib import Path

from named_repair_eval import MODELS, collect, prepare
from repair_pilot import build_scenes, generate, reference_response


class NamedRepairEvalTests(unittest.TestCase):
    def test_prepare_collect_and_input_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, output = root / "dataset", root / "trials"
            generate(dataset)
            manifest = prepare(dataset, output)
            self.assertEqual(manifest["trial_count"], 24)
            self.assertEqual(manifest["models"], list(MODELS))
            answers = {entry["example_id"]: entry
                       for entry in reference_response(build_scenes())["examples"]}
            for trial in manifest["trials"]:
                request = json.loads(Path(trial["request"]).read_text())
                self.assertEqual(len(request["examples"]), 1)
                self.assertEqual(request["examples"][0]["example_id"], trial["example_id"])
                Path(trial["response"]).write_text(json.dumps(
                    {"examples": [answers[trial["example_id"]]]}))
            summary = collect(dataset, output)
            self.assertEqual(summary, collect(dataset, output))
            for model in MODELS:
                self.assertTrue(all(value == 1 for value in summary["models"][model]["accuracy"].values()))
                self.assertTrue(all(pair["cosmetic_invariance"]
                                    for pair in summary["models"][model]["matched_pairs"]))
            with self.assertRaises(ValueError):
                prepare(dataset, output)
            first = manifest["trials"][0]
            Path(first["response"]).unlink()
            with self.assertRaisesRegex(ValueError, "Missing trial response"):
                collect(dataset, output)
            Path(first["response"]).write_text(json.dumps({"examples": [answers[first["example_id"]]]}))
            request = json.loads(Path(first["request"]).read_text())
            Path(request["examples"][0]["image"]).write_bytes(b"changed image")
            with self.assertRaisesRegex(ValueError, "Image changed"):
                collect(dataset, output)


if __name__ == "__main__":
    unittest.main()
