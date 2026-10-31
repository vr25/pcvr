"""Run local open-weight vision-language models (via Ollama) on the repair-pilot scenes.

Each call sees only one public request and its image. Raw model text is archived.
Scoring derives valid and cheapest sets from the model's own per-candidate
violations (the canonical answer), and separately reports the sets the model
stated directly, so internal inconsistency is measured rather than hidden.
"""

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path

from repair_pilot import ROOM, VIOLATIONS, score

OLLAMA = "http://localhost:11434/api/chat"
PROMPT = """{scene_intro}

Goal: {goal}
Plan: {plan}
Preservation requirements: {preserve}

Rules:
{assumptions}

Instructions: {instructions}

Candidate final trolley placements (dashed boxes labelled A-F) with public costs:
{candidates}

For EVERY candidate, list ALL violated categories from: target, destination,
exit, furniture, boundary. Use [] when a candidate violates none.
valid_interventions = candidates with no violations.
minimum_cost_interventions = the lowest-cost members of the valid set.
no_solution = true exactly when no candidate is valid.

Answer with JSON only:
{{"candidate_violations": {{"A": [...], ...}}, "valid_interventions": [...],
"minimum_cost_interventions": [...], "no_solution": false}}
"""


IMAGE_INTRO = "You are shown an overhead diagram of a scene and a task."
TEXT_INTRO = """No image is provided. The overhead diagram is described exactly in coordinates.
Rectangles are (left, top, right, bottom) in diagram units; y grows downward.
{geometry}"""


def rect(r):
    return f"({r['left']}, {r['top']}, {r['right']}, {r['bottom']})"


def geometry(scene):
    """Describe only drawn geometry; no labels or evaluations are included."""
    lines = [f"Room interior: {ROOM}"]
    lines += [f"{name.upper()} zone (blue dotted): {rect(zone)}" for name, zone in scene["zones"].items()]
    lines += [f"Furniture {i + 1}: {rect(f)}" for i, f in enumerate(scene["furniture"])]
    lines.append(f"Trolley initial position: {rect(scene['initial'])}")
    lines += [f"Candidate {c['candidate_id']} footprint: {rect(c['footprint'])}" for c in scene["candidates"]]
    return "\n".join(lines)


def build_prompt(example, public, scene=None):
    intro = IMAGE_INTRO if scene is None else TEXT_INTRO.format(geometry=geometry(scene))
    return PROMPT.format(
        scene_intro=intro,
        goal=example["goal"],
        plan="; ".join(example["plan"]),
        preserve="; ".join(example["preservation_requirements"]),
        assumptions="\n".join(f"- {line}" for line in public["assumptions"]),
        instructions=public["instructions"],
        candidates="\n".join(f"{c['candidate_id']}: cost {c['cost']}" for c in example["candidates"]),
    )


def schema(ids):
    letters = {"type": "array", "items": {"type": "string", "enum": ids}}
    return {
        "type": "object",
        "properties": {
            "candidate_violations": {
                "type": "object",
                "properties": {i: {"type": "array", "items": {"type": "string", "enum": sorted(VIOLATIONS)}}
                               for i in ids},
                "required": ids,
            },
            "valid_interventions": letters,
            "minimum_cost_interventions": letters,
            "no_solution": {"type": "boolean"},
        },
        "required": ["candidate_violations", "valid_interventions",
                     "minimum_cost_interventions", "no_solution"],
    }


def call(model, prompt, image, ids, temperature, seed):
    message = {"role": "user", "content": prompt}
    if image is not None:
        message["images"] = [base64.b64encode(image.read_bytes()).decode()]
    body = {
        "model": model, "stream": False, "format": schema(ids),
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192},
        "messages": [message],
    }
    request = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())["message"]["content"]


def normalize(raw, example):
    """Return (derived entry, stated sets, parse_ok)."""
    ids = [c["candidate_id"] for c in example["candidates"]]
    costs = {c["candidate_id"]: c["cost"] for c in example["candidates"]}
    try:
        data = json.loads(raw)
        parse_ok = isinstance(data, dict)
    except json.JSONDecodeError:
        data, parse_ok = {}, False
    data = data if isinstance(data, dict) else {}
    given = data.get("candidate_violations") if isinstance(data.get("candidate_violations"), dict) else {}
    violations = {}
    for i in ids:
        values = given.get(i, [])
        values = values if isinstance(values, list) else []
        violations[i] = sorted({v for v in values if v in VIOLATIONS})
    valid = sorted(i for i in ids if not violations[i])
    low = min((costs[i] for i in valid), default=None)
    entry = {
        "example_id": example["example_id"], "candidate_violations": violations,
        "valid_interventions": valid,
        "minimum_cost_interventions": sorted(i for i in valid if costs[i] == low),
        "no_solution": not valid,
    }

    def letters(key):
        values = data.get(key)
        return sorted({v for v in values if v in ids}) if isinstance(values, list) else None

    stated = {"valid_interventions": letters("valid_interventions"),
              "minimum_cost_interventions": letters("minimum_cost_interventions"),
              "no_solution": data.get("no_solution") if isinstance(data.get("no_solution"), bool) else None}
    return entry, stated, parse_ok


def run(args):
    public = json.loads((args.dataset / "examples.public.json").read_text())
    scenes = {}
    if args.text_only:
        private = json.loads((args.dataset / "examples.private.json").read_text())
        scenes = {e["example_id"]: e["scene"] for e in private["examples"]}
    for model in args.models:
        for rep in range(args.repeats):
            temperature = 0.0 if rep == 0 else args.temperature
            out = args.output / model.replace(":", "_").replace("/", "_") / f"rep{rep}"
            out.mkdir(parents=True, exist_ok=True)
            for example in public["examples"]:
                path = out / f"{example['example_id']}.json"
                if path.exists():
                    continue
                ids = [c["candidate_id"] for c in example["candidates"]]
                start = time.time()
                scene = scenes.get(example["example_id"])
                image = None if args.text_only else args.dataset / example["image"]
                raw = call(model, build_prompt(example, public, scene), image, ids, temperature, seed=rep)
                path.write_text(json.dumps({
                    "model": model, "replicate": rep, "temperature": temperature, "seed": rep,
                    "example_id": example["example_id"], "seconds": round(time.time() - start, 1),
                    "raw": raw}, indent=2) + "\n")
                print(model, rep, example["example_id"], f"{time.time() - start:.0f}s", flush=True)


def evaluate(args):
    public = json.loads((args.dataset / "examples.public.json").read_text())
    private = json.loads((args.dataset / "examples.private.json").read_text())
    truth = {e["example_id"]: e for e in private["examples"]}
    summary = {}
    for model_dir in sorted(p for p in args.output.iterdir() if p.is_dir()):
        reps, sets = {}, {}
        for rep_dir in sorted(model_dir.glob("rep*")):
            files = {p.stem: json.loads(p.read_text()) for p in rep_dir.glob("*.json")}
            if len(files) != len(public["examples"]):
                continue
            entries, consistent, stated_full, stated_min, parsed = [], 0, 0, 0, 0
            for example in public["examples"]:
                entry, stated, ok = normalize(files[example["example_id"]]["raw"], example)
                entries.append(entry)
                parsed += ok
                gold = truth[example["example_id"]]
                consistent += (stated["valid_interventions"] == entry["valid_interventions"]
                               and stated["minimum_cost_interventions"] == entry["minimum_cost_interventions"]
                               and stated["no_solution"] == entry["no_solution"])
                stated_full += stated["valid_interventions"] == sorted(gold["valid_interventions"])
                stated_min += stated["minimum_cost_interventions"] == sorted(gold["minimum_cost_interventions"])
            result = score({"examples": entries}, public, private)
            n = len(entries)
            result["stated_sets"] = {"internally_consistent": consistent / n, "parsed": parsed / n,
                                     "full_repair_set_exact": stated_full / n,
                                     "minimum_set_exact": stated_min / n}
            reps[rep_dir.name] = result
            sets[rep_dir.name] = {e["example_id"]: e["valid_interventions"] for e in entries}
        if reps:
            ids = [e["example_id"] for e in public["examples"]]
            stable = sum(len({tuple(s[i]) for s in sets.values()}) == 1 for i in ids)
            summary[model_dir.name] = {"replicates": reps,
                                       "valid_set_identical_across_runs": stable / len(ids)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for model, data in summary.items():
        print(f"{model}: valid set identical across runs = {data['valid_set_identical_across_runs']:.2f}")
        for rep, r in data["replicates"].items():
            a, s = r["accuracy"], r["stated_sets"]
            print(f"{model:22s} {rep}  full={a['full_repair_set_exact']:.2f} min={a['minimum_set_exact']:.2f} "
                  f"viol={a['candidate_violation_set_exact']:.2f} prec={r['valid_repair_precision']} "
                  f"FP={r['counts']['false_positive']} | stated full={s['full_repair_set_exact']:.2f} "
                  f"min={s['minimum_set_exact']:.2f} consistent={s['internally_consistent']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "evaluate"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/repair-pilot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/repair-pilot-local"))
    parser.add_argument("--models", nargs="+", default=["qwen2.5vl:3b", "gemma3:4b"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--text-only", action="store_true",
                        help="Replace the image with a coordinate description of the drawn geometry.")
    args = parser.parse_args()
    run(args) if args.command == "run" else evaluate(args)


if __name__ == "__main__":
    main()
