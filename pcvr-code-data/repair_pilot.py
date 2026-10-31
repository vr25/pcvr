"""A bounded, illustrated clearance-repair pilot, not a physics simulator."""

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1280, 960
ROOM = (60, 100, 1220, 850)
OBJECT_SIZE = (120, 60)
VIOLATIONS = ("target", "destination", "exit", "furniture", "boundary")


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self):
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("Rectangles must have positive area.")

    def overlaps(self, other):
        return (max(self.left, other.left) < min(self.right, other.right)
                and max(self.top, other.top) < min(self.bottom, other.bottom))

    def contains(self, other):
        return (self.left <= other.left and self.top <= other.top
                and self.right >= other.right and self.bottom >= other.bottom)

    def box(self):
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    footprint: Rect
    cost: int


@dataclass(frozen=True)
class Scene:
    example_id: str
    family: str
    variant: str
    target_object: str
    destination_name: str
    initial: Rect
    zones: dict[str, Rect]
    furniture: tuple[Rect, ...]
    candidates: tuple[Candidate, ...]
    accent: str


def transform(rect, family):
    left, top, right, bottom = rect.box()
    if family in ("workshop", "storage"):
        left, right = WIDTH - right, WIDTH - left
    if family == "storage":
        top, bottom = 950 - bottom, 950 - top
    return Rect(left, top, right, bottom)


def build_scenes():
    scenes = []
    for family, target, destination in (
        ("kitchen", "orange mug", "countertop"),
        ("workshop", "orange drill", "workbench"),
        ("storage", "orange parcel", "packing table"),
    ):
        for variant in ("base", "cosmetic", "blocked-near", "no-room"):
            initial = transform(Rect(180, 265, 300, 325), family)
            zones = {key: transform(rect, family) for key, rect in {
                "target": Rect(100, 240, 390, 340),
                "destination": Rect(840, 240, 1140, 340),
                "exit": Rect(950, 640, 1140, 795),
            }.items()}
            furniture = [
                Rect(100, 120, 390, 235), Rect(840, 120, 1140, 235),
                Rect(100, 430, 340, 650), Rect(700, 510, 890, 645),
            ]
            if variant in ("blocked-near", "no-room"):
                furniture.append(Rect(400, 410, 555, 500))
            if variant == "no-room":
                furniture.append(Rect(580, 410, 725, 500))
            positions = [(180, 265), (880, 265), (980, 690),
                         (410, 420), (590, 420), (720, 580)]
            offset = {"kitchen": 0, "workshop": 2, "storage": 4}[family]
            candidates = []
            for index, (x, y) in enumerate(positions):
                footprint = transform(Rect(x, y, x + 120, y + 60), family)
                candidates.append(Candidate(
                    chr(ord("A") + (index + offset) % len(positions)),
                    footprint,
                    abs(footprint.left - initial.left)
                    + abs(footprint.top - initial.top),
                ))
            example_id = hashlib.sha256(f"{family}:{variant}".encode()).hexdigest()[:16]
            scenes.append(Scene(
                example_id, family, variant, target, destination, initial,
                zones, tuple(transform(rect, family) for rect in furniture),
                tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id)),
                "#537e76" if variant != "cosmetic" else "#7973a1",
            ))
    return scenes


def diagnose(scene, footprint):
    failures = [key for key, zone in scene.zones.items() if footprint.overlaps(zone)]
    if any(footprint.overlaps(rect) for rect in scene.furniture):
        failures.append("furniture")
    if not Rect(*ROOM).contains(footprint):
        failures.append("boundary")
    return failures


def oracle(scene):
    evaluations = []
    for candidate in scene.candidates:
        failures = diagnose(scene, candidate.footprint)
        evaluations.append({
            "candidate_id": candidate.candidate_id,
            "cost": candidate.cost,
            "violations": failures,
            "valid": not failures,
        })
    valid = [entry for entry in evaluations if entry["valid"]]
    minimum = min((entry["cost"] for entry in valid), default=None)
    return {
        "example_id": scene.example_id,
        "valid_interventions": [entry["candidate_id"] for entry in valid],
        "minimum_cost_interventions": [
            entry["candidate_id"] for entry in valid if entry["cost"] == minimum
        ],
        "minimum_cost": minimum,
        "no_solution": not valid,
        "candidate_evaluations": evaluations,
        "initial_violations": diagnose(scene, scene.initial),
    }


def font(size):
    return ImageFont.load_default(size=size)


def render(scene):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f5f0e6")
    draw = ImageDraw.Draw(image)
    draw.text((60, 25), f"{scene.family.title()} | overhead clearance study",
              fill="#293f3a", font=font(30))
    draw.text((60, 66), "Dotted zones must stay clear. Dashed A-F boxes are candidate placements.",
              fill="#52675e", font=font(19))
    draw.rectangle(ROOM, fill="#dec9aa", outline="#53655c", width=8)
    for y in range(110, 850, 40):
        draw.line((66, y, 1213, y), fill="#cdb695", width=2)
        for x in range(66 + (y // 40 % 2) * 120, 1200, 240):
            draw.line((x, y, x, min(y + 40, 844)), fill="#cdb695", width=2)

    def dashed(rect, color, width=2, dash=9):
        left, top, right, bottom = rect.box()
        for x in range(left, right, dash * 2):
            draw.line((x, top, min(x + dash, right - 1), top), fill=color, width=width)
            draw.line((x, bottom - 1, min(x + dash, right - 1), bottom - 1),
                      fill=color, width=width)
        for y in range(top, bottom, dash * 2):
            draw.line((left, y, left, min(y + dash, bottom - 1)), fill=color, width=width)
            draw.line((right - 1, y, right - 1, min(y + dash, bottom - 1)),
                      fill=color, width=width)

    def furnished(rect, index):
        left, top, right, bottom = rect.box()
        draw.rounded_rectangle((left + 7, top + 7, right + 7, bottom + 7),
                               radius=8, fill="#b3a087")
        draw.rounded_rectangle((left, top, right - 1, bottom - 1), radius=8,
                               fill=scene.accent if index < 2 else "#a58259",
                               outline="#435f55", width=3)
        for x in range(left + 8, right - 10, 65):
            draw.line((x, top + 9, x, bottom - 10), fill="#bfc5ab", width=2)
            draw.rounded_rectangle((x + 9, bottom - 18, min(x + 42, right - 8), bottom - 12),
                                   radius=2, fill="#e8debb")
        if index == 0:
            cx, cy = (left + right) // 2, (top + bottom) // 2
            if scene.family == "kitchen":
                draw.ellipse((cx - 21, cy - 21, cx + 21, cy + 21),
                             fill="#da8247", outline="#8e4a2c", width=3)
                draw.arc((cx + 12, cy - 14, cx + 36, cy + 14), -90, 90,
                         fill="#8e4a2c", width=5)
            elif scene.family == "workshop":
                draw.rounded_rectangle((cx - 30, cy - 18, cx + 19, cy + 10),
                                       radius=5, fill="#da8247")
                draw.rectangle((cx - 11, cy + 8, cx + 3, cy + 27), fill="#da8247")
                draw.rectangle((cx + 19, cy - 10, cx + 39, cy - 3), fill="#465551")
            else:
                draw.rectangle((cx - 30, cy - 22, cx + 30, cy + 22),
                               fill="#da8247", outline="#8e4a2c", width=3)
                draw.line((cx, cy - 22, cx, cy + 22), fill="#efcd8e", width=7)
            draw.text((left + 10, top + 9), scene.target_object,
                      fill="#fff8e7", font=font(17))
        elif index == 1:
            draw.text((left + 10, top + 9), scene.destination_name,
                      fill="#fff8e7", font=font(17))
        else:
            for x in range(left + 18, right - 25, 48):
                draw.rectangle((x, top + 24, x + 26, top + 48),
                               fill="#d9c7a5", outline="#806344", width=2)

    for index, rect in enumerate(scene.furniture):
        furnished(rect, index)
    for key, zone in scene.zones.items():
        dashed(zone, "#496b80", width=3, dash=5)
        draw.text((zone.left + 5, zone.top + 5), key.upper(),
                  fill="#314d60", font=font(17))

    # Only the footprint, not the wheels or surface decoration, enters the checker.
    left, top, right, bottom = scene.initial.box()
    draw.rectangle((left, top, right - 1, bottom - 1),
                   fill="#bb905c", outline="#725338", width=3)
    for x in range(left + 14, right - 10, 23):
        draw.line((x, top + 6, x, bottom - 7), fill="#e4c498", width=2)
    for x in (left + 7, right - 17):
        for y in (top + 7, bottom - 17):
            draw.ellipse((x, y, x + 9, y + 9), fill="#384b45")
    draw.text((left + 16, top + 21), "TROLLEY", fill="#483d30", font=font(15))
    for candidate in scene.candidates:
        dashed(candidate.footprint, "#463e5d", width=3)
        draw.rectangle((candidate.footprint.left + 4, candidate.footprint.top + 4,
                        candidate.footprint.left + 28, candidate.footprint.top + 28),
                       fill="#fbf7ed", outline="#463e5d")
        draw.text((candidate.footprint.left + 9, candidate.footprint.top + 6),
                  candidate.candidate_id, fill="#463e5d", font=font(17))
    draw.text((60, 880), "Move the trolley to one marked footprint; do not rotate it.",
              fill="#314e45", font=font(23))
    draw.text((60, 917), "Formal 2D access rules only: no door physics, grasping, or relocation path model.",
              fill="#687168", font=font(18))
    return image


def public_example(scene, pixels):
    return {
        "example_id": scene.example_id,
        "family": scene.family,
        "image": f"images/{scene.example_id}.png",
        "sha256": hashlib.sha256(pixels).hexdigest(),
        "goal": f"Retrieve the {scene.target_object} and put it on the {scene.destination_name}.",
        "plan": ["Access the target fixture", "Retrieve the target object",
                 "Place it on the destination surface"],
        "preservation_requirements": ["Keep destination access clear", "Keep the exit lane clear"],
        "candidates": [{"candidate_id": c.candidate_id, "cost": c.cost}
                       for c in scene.candidates],
    }


def validate_response(response, public):
    if not isinstance(response, dict) or set(response) != {"examples"}:
        raise ValueError("Response must contain exactly an examples list.")
    if not isinstance(response["examples"], list):
        raise ValueError("examples must be a list.")
    expected = {example["example_id"]: example for example in public["examples"]}
    received = {}
    fields = {"example_id", "valid_interventions", "minimum_cost_interventions",
              "no_solution", "candidate_violations"}
    for entry in response["examples"]:
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(f"Each response must contain exactly {sorted(fields)}.")
        identifier = entry["example_id"]
        if not isinstance(identifier, str) or identifier not in expected or identifier in received:
            raise ValueError("Unknown or duplicate example_id.")
        ids = {c["candidate_id"] for c in expected[identifier]["candidates"]}
        for key in ("valid_interventions", "minimum_cost_interventions"):
            values = entry[key]
            if (not isinstance(values, list)
                    or any(not isinstance(value, str) or value not in ids for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError(f"{key} must contain unique, known candidate IDs.")
        if type(entry["no_solution"]) is not bool:
            raise ValueError("no_solution must be a boolean.")
        violations = entry["candidate_violations"]
        if not isinstance(violations, dict) or set(violations) != ids:
            raise ValueError("candidate_violations must cover every candidate.")
        for values in violations.values():
            if (not isinstance(values, list)
                    or any(not isinstance(value, str) or value not in VIOLATIONS for value in values)
                    or len(set(values)) != len(values)):
                raise ValueError("Violation lists must contain unique, permitted categories.")
        derived = {key for key, values in violations.items() if not values}
        costs = {c["candidate_id"]: c["cost"] for c in expected[identifier]["candidates"]}
        minimum = min((costs[key] for key in derived), default=None)
        cheapest = {key for key in derived if costs[key] == minimum}
        if (set(entry["valid_interventions"]) != derived
                or set(entry["minimum_cost_interventions"]) != cheapest
                or entry["no_solution"] != (not derived)):
            raise ValueError("Validity, costs, and violations must be internally consistent.")
        received[identifier] = entry
    if set(received) != set(expected):
        raise ValueError("Response must cover every example exactly once.")
    return received


def score(response, public, private):
    received = validate_response(response, public)
    totals = dict(full_repair_set_exact=0, minimum_set_exact=0,
                  no_solution_correct=0, candidate_violation_set_exact=0)
    tp = fp = fn = 0
    by_id = {}
    for truth in private["examples"]:
        predicted = received[truth["example_id"]]
        actual_set, predicted_set = set(truth["valid_interventions"]), set(predicted["valid_interventions"])
        tp += len(actual_set & predicted_set)
        fp += len(predicted_set - actual_set)
        fn += len(actual_set - predicted_set)
        checks = {
            "full_repair_set_exact": actual_set == predicted_set,
            "minimum_set_exact": set(truth["minimum_cost_interventions"])
            == set(predicted["minimum_cost_interventions"]),
            "no_solution_correct": truth["no_solution"] == predicted["no_solution"],
            "candidate_violation_set_exact": all(
                set(entry["violations"]) == set(predicted["candidate_violations"][entry["candidate_id"]])
                for entry in truth["candidate_evaluations"]),
        }
        by_id[truth["example_id"]] = checks
        for key, passed in checks.items():
            totals[key] += passed
    count = len(private["examples"])
    if count == 0:
        raise ValueError("Cannot score an empty dataset.")
    return {
        "examples": count,
        "accuracy": {key: value / count for key, value in totals.items()},
        "valid_repair_precision": tp / (tp + fp) if tp + fp else None,
        "valid_repair_recall": tp / (tp + fn) if tp + fn else None,
        "counts": {"true_positive": tp, "false_positive": fp, "false_negative": fn},
        "per_example": by_id,
    }


def reference_response(scenes, local_only=False):
    entries = []
    for scene in scenes:
        truth = oracle(scene)
        violations = {
            entry["candidate_id"]: [v for v in entry["violations"]
                                    if not local_only or v not in ("destination", "exit")]
            for entry in truth["candidate_evaluations"]
        }
        valid = [key for key, values in violations.items() if not values]
        costs = {candidate.candidate_id: candidate.cost for candidate in scene.candidates}
        minimum = min((costs[key] for key in valid), default=None)
        entries.append({
            "example_id": scene.example_id,
            "valid_interventions": valid,
            "minimum_cost_interventions": [key for key in valid if costs[key] == minimum],
            "no_solution": not valid,
            "candidate_violations": violations,
        })
    return {"examples": entries}


def build_artifacts():
    scenes = build_scenes()
    files, images, examples, truths = {}, [], [], []
    for scene in scenes:
        image = render(scene)
        images.append(image)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        pixels = stream.getvalue()
        files[f"images/{scene.example_id}.png"] = pixels
        examples.append(public_example(scene, pixels))
        truths.append({**oracle(scene), "scene": asdict(scene)})
    public = {
        "schema_version": 1,
        "instructions": (
            "Identify all valid final trolley placements and all cheapest valid placements. "
            "Report every candidate's violations using target, destination, exit, furniture, boundary. "
            "Report no_solution when none is valid. A candidate can violate several constraints."
        ),
        "assumptions": [
            "Overhead illustrations encode a 2D, axis-aligned rectangle world.",
            "Blue dotted TARGET, DESTINATION and EXIT zones are exact operational clearance regions.",
            "All three zones must be clear after relocation; furniture must also remain clear.",
            "Dashed A-F boxes show the entire final trolley footprint; the drawn trolley is its initial state.",
            "Strict positive-area overlap is a violation; boundary contact is allowed.",
            "All furniture is fixed. Only the trolley moves, without rotation; staying costs zero.",
            "Relocation is an abstract placement action: its transit path is not evaluated.",
            "Cost is Manhattan displacement in diagram units, not measured physical effort.",
            "Grasping, door sweeps, dynamics and visibility are outside this formal task.",
        ],
        "response_fields": ["example_id", "valid_interventions", "minimum_cost_interventions",
                            "no_solution", "candidate_violations"],
        "examples": examples,
    }
    private = {"schema_version": 1, "examples": truths}
    sheet = Image.new("RGB", (1280, 960), "#f5f0e6")
    for index, image in enumerate(images):
        sheet.paste(image.resize((426, 240)), ((index % 3) * 426, (index // 3) * 240))
    stream = io.BytesIO()
    sheet.save(stream, format="PNG")
    files["contact-sheet.png"] = stream.getvalue()
    report = {
        "scene_count": len(scenes), "candidate_count": sum(len(s.candidates) for s in scenes),
        "families": ["kitchen", "workshop", "storage"],
        "solvable_examples": sum(not truth["no_solution"] for truth in truths),
        "no_solution_examples": sum(truth["no_solution"] for truth in truths),
        "matched_pairs": [
            {"base": group["base"], "cosmetic": group["cosmetic"],
             "blocked_near": group["blocked-near"], "no_room": group["no-room"]}
            for family in ("kitchen", "workshop", "storage")
            for group in [{s.variant: s.example_id for s in scenes if s.family == family}]
        ],
        "oracle_reference": score(reference_response(scenes), public, private),
        "local_only_reference": score(reference_response(scenes, local_only=True), public, private),
        "learned_model_results_included": False,
        "limitations": [
            "Designed examples, not independent samples or a train/test split.",
            "Families reuse mirrored constraint layouts: this does not test structural transfer.",
            "Labels establish only the declared rectangle-clearance model, not real-world executability.",
            "Marked clearance regions simplify perception; rich unmarked imagery remains future work.",
            "Reference policies are programmed rules, not measurements of VLM ability.",
            "No claim of task novelty or demonstrated frontier-model failure.",
        ],
    }
    for name, data in (("examples.public.json", public), ("examples.private.json", private),
                       ("report.json", report)):
        files[name] = (json.dumps(data, indent=2) + "\n").encode()
    return files


def generate(output):
    files = build_artifacts()
    if output.is_symlink():
        raise ValueError("Refusing a symlink output directory.")
    if output.exists():
        existing = {path.relative_to(output).as_posix(): path for path in output.rglob("*")
                    if path.is_file() or path.is_symlink()}
        if (set(existing) != set(files)
                or any(path.is_symlink() or path.read_bytes() != files[name]
                       for name, path in existing.items())
                or any(path.is_symlink() for path in output.rglob("*"))):
            raise ValueError("Output differs from this pilot; choose a fresh output directory.")
        return
    output.mkdir(parents=True)
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--output", type=Path, default=Path("artifacts/repair-pilot"))
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--dataset", type=Path, default=Path("artifacts/repair-pilot"))
    score_parser.add_argument("--responses", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output)
        print(f"Saved 12 scenes / 72 candidate placements to {args.output.resolve()}")
    else:
        public = json.loads((args.dataset / "examples.public.json").read_text())
        private = json.loads((args.dataset / "examples.private.json").read_text())
        response = json.loads(args.responses.read_text())
        print(json.dumps(score(response, public, private), indent=2))


if __name__ == "__main__":
    main()
