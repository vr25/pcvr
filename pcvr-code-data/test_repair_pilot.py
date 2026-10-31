import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from repair_pilot import (
    ROOM, Rect, build_artifacts, build_scenes, diagnose, generate, oracle,
    reference_response, score,
)


class RepairPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes = build_scenes()
        cls.files = build_artifacts()
        cls.public = json.loads(cls.files["examples.public.json"])
        cls.private = json.loads(cls.files["examples.private.json"])

    def test_composition_and_independent_occupancy_oracle(self):
        self.assertEqual(len(self.scenes), 12)
        self.assertEqual(sum(len(scene.candidates) for scene in self.scenes), 72)
        for scene in self.scenes:
            self.assertEqual(diagnose(scene, scene.initial), ["target"])
            self.assertEqual(len({c.candidate_id for c in scene.candidates}), 6)
            for candidate in scene.candidates:
                left, top, right, bottom = candidate.footprint.box()
                cells = {(x, y) for x in range(left, right) for y in range(top, bottom)}
                expected = []
                for key, zone in scene.zones.items():
                    if any(zone.left <= x < zone.right and zone.top <= y < zone.bottom
                           for x, y in cells):
                        expected.append(key)
                if any(any(rect.left <= x < rect.right and rect.top <= y < rect.bottom
                           for x, y in cells) for rect in scene.furniture):
                    expected.append("furniture")
                if any(not (ROOM[0] <= x < ROOM[2] and ROOM[1] <= y < ROOM[3])
                       for x, y in cells):
                    expected.append("boundary")
                self.assertEqual(diagnose(scene, candidate.footprint), expected)
                self.assertEqual((right - left, bottom - top), (120, 60))
                self.assertEqual(candidate.cost, abs(left - scene.initial.left)
                                 + abs(top - scene.initial.top))

    def test_matched_contrasts_and_minimum_certificates(self):
        for family in ("kitchen", "workshop", "storage"):
            group = {scene.variant: oracle(scene) for scene in self.scenes if scene.family == family}
            for key in ("valid_interventions", "minimum_cost_interventions", "candidate_evaluations"):
                self.assertEqual(group["base"][key], group["cosmetic"][key])
            self.assertEqual(len(group["base"]["valid_interventions"]), 2)
            self.assertEqual(len(group["blocked-near"]["valid_interventions"]), 1)
            self.assertNotEqual(group["base"]["minimum_cost_interventions"],
                                group["blocked-near"]["minimum_cost_interventions"])
            self.assertTrue(group["no-room"]["no_solution"])
            self.assertEqual(group["no-room"]["minimum_cost_interventions"], [])
            self.assertIsNone(group["no-room"]["minimum_cost"])
        for scene in self.scenes:
            truth = oracle(scene)
            feasible = [candidate for candidate in scene.candidates
                        if not diagnose(scene, candidate.footprint)]
            expected = [candidate.candidate_id for candidate in feasible
                        if candidate.cost == min(c.cost for c in feasible)]
            self.assertEqual(truth["minimum_cost_interventions"], expected)

    def test_contact_is_allowed_and_boundary_violation(self):
        self.assertFalse(Rect(0, 0, 10, 10).overlaps(Rect(10, 0, 20, 10)))
        self.assertTrue(Rect(0, 0, 10, 10).overlaps(Rect(9, 0, 20, 10)))
        self.assertIn("boundary", diagnose(self.scenes[0], Rect(0, 0, 120, 60)))
        with self.assertRaises(ValueError):
            Rect(0, 0, 0, 1)

    def test_saved_pixels_public_separation_and_determinism(self):
        self.assertEqual(self.files, build_artifacts())
        self.assertEqual(len([name for name in self.files if name.startswith("images/")]), 12)
        for scene, example in zip(self.scenes, self.public["examples"]):
            self.assertEqual(set(example), {
                "example_id", "family", "image", "sha256", "goal", "plan",
                "preservation_requirements", "candidates",
            })
            pixels = self.files[example["image"]]
            self.assertEqual(hashlib.sha256(pixels).hexdigest(), example["sha256"])
            with Image.open(io.BytesIO(pixels)) as image:
                self.assertEqual(image.size, (1280, 960))
                self.assertEqual(image.info, {})
                for candidate in scene.candidates:
                    # Candidate corner outlines come directly from verified geometry.
                    self.assertEqual(image.getpixel((candidate.footprint.left, candidate.footprint.top)),
                                     (70, 62, 93))
        for field in ('"valid_interventions"', '"minimum_cost_interventions"', '"variant"', '"scene"'):
            self.assertNotIn(field + ":", json.dumps(self.public))

    def test_score_references_and_response_validation(self):
        oracle_result = score(reference_response(self.scenes), self.public, self.private)
        self.assertTrue(all(value == 1 for value in oracle_result["accuracy"].values()))
        local = score(reference_response(self.scenes, local_only=True), self.public, self.private)
        self.assertEqual(local["accuracy"]["full_repair_set_exact"], 0)
        self.assertEqual(local["accuracy"]["minimum_set_exact"], 0.75)
        self.assertEqual(local["accuracy"]["no_solution_correct"], 0.75)
        self.assertEqual(local["valid_repair_recall"], 1)
        response = reference_response(self.scenes)
        response["examples"].pop()
        with self.assertRaises(ValueError):
            score(response, self.public, self.private)
        response = reference_response(self.scenes)
        response["examples"].append(copy.deepcopy(response["examples"][0]))
        with self.assertRaises(ValueError):
            score(response, self.public, self.private)
        response = reference_response(self.scenes)
        response["examples"][0]["no_solution"] = True
        with self.assertRaises(ValueError):
            score(response, self.public, self.private)
        response = reference_response(self.scenes)
        response["examples"][0]["valid_interventions"] = ["UNKNOWN"]
        with self.assertRaises(ValueError):
            score(response, self.public, self.private)

    def test_safe_idempotent_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "pilot"
            generate(output)
            generate(output)
            sentinel = output / "user-note.txt"
            sentinel.write_text("Keep this file.")
            with self.assertRaises(ValueError):
                generate(output)
            self.assertEqual(sentinel.read_text(), "Keep this file.")
            link = root / "linked"
            link.symlink_to(output, target_is_directory=True)
            with self.assertRaises(ValueError):
                generate(link)


if __name__ == "__main__":
    unittest.main()
