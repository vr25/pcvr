"""Prepare isolated Copilot vision trials and collect their unedited decisions."""

import argparse
import hashlib
import json
from pathlib import Path

from repair_pilot import score, validate_response


MODELS = ("gpt-6.1-sol", "claude-opus-5.5")
SOLVER_PROMPT = """You are a vision-language solver in an experimental trial.
Read ONLY this trial's request.json and the ONE image it lists. Do not read
other trials, dataset-wide metadata, generator code, tests, README, private
labels, reports, history, or session artifacts. Do not delegate.
Use the image viewing tool on the full-resolution PNG. Make visual judgments;
do not use scripts to extract pixels, geometry, or answers. Read the public
rules carefully. This is a fresh trial: no other scene should enter this context.
Do not infer 3D physics beyond the declared 2D clearance task.

Save your decisions using apply_patch to this trial's response.json. Return
exactly {"examples": [ENTRY]}, with exactly one ENTRY containing:
example_id, valid_interventions, minimum_cost_interventions, no_solution,
candidate_violations. Copy example_id from the request. Lists contain unique
candidate letters. no_solution is a boolean. candidate_violations maps EVERY
candidate letter to a list of ALL violated categories: target, destination,
exit, furniture, boundary. Use [] when a candidate violates none.
Your valid_interventions must be exactly the candidates with no violations.
Your minimum_cost_interventions must be all lowest-public-cost members of
that valid set. no_solution must be true exactly when the valid set is empty.
No alternative moves, rotations, explanations, markdown, or extra JSON keys.
Do not inspect gold or change predictions after any evaluation feedback.
STOP after writing response.json and briefly confirming the path.
"""


def dump(data):
    return json.dumps(data, indent=2) + "\n"


def prepare(dataset, output):
    if output.exists() or output.is_symlink():
        raise ValueError("Trial directory already exists; choose a fresh path.")
    dataset = dataset.resolve()
    public = json.loads((dataset / "examples.public.json").read_text())
    trials = []
    output.mkdir(parents=True)
    for model in MODELS:
        for example in public["examples"]:
            image = (dataset / example["image"]).resolve()
            if not image.is_relative_to(dataset):
                raise ValueError("Task image must remain inside the dataset.")
            if hashlib.sha256(image.read_bytes()).hexdigest() != example["sha256"]:
                raise ValueError(f"Image hash mismatch: {image}")
            trial = output / model / example["example_id"]
            trial.mkdir(parents=True)
            request = {key: value for key, value in public.items() if key != "examples"}
            request["examples"] = [{**example, "image": str(image)}]
            (trial / "request.json").write_text(dump(request))
            prompt = SOLVER_PROMPT + f"\nTrial directory: {trial.resolve()}\n"
            (trial / "prompt.txt").write_text(prompt)
            trials.append({
                "configured_model_id": model,
                "example_id": example["example_id"],
                "prompt": str((trial / "prompt.txt").resolve()),
                "request": str((trial / "request.json").resolve()),
                "response": str((trial / "response.json").resolve()),
                "image_sha256": example["sha256"],
            })
    manifest = {
        "models": list(MODELS), "trial_count": len(trials),
        "method": "One new Copilot task agent per image; explicit model selection.",
        "blinding": "Instruction-based; not a filesystem sandbox.",
        "settings": "Harness defaults; temperature and provider sampling controls not exposed.",
        "runtime_model_resolution": "Configured IDs archived; provider-resolved versions not exposed.",
        "prompt_archive": "Trial instructions and public requests; harness system prompts not archived.",
        "trials": trials,
    }
    (output / "manifest.json").write_text(dump(manifest))
    return manifest


def collect(dataset, output):
    public = json.loads((dataset / "examples.public.json").read_text())
    private = json.loads((dataset / "examples.private.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    results = {}
    for model in MODELS:
        entries = []
        trials = [trial for trial in manifest["trials"] if trial["configured_model_id"] == model]
        if len(trials) != len(public["examples"]):
            raise ValueError(f"Incomplete manifest for {model}.")
        for trial in trials:
            request = json.loads(Path(trial["request"]).read_text())
            example = request["examples"][0]
            if hashlib.sha256(Path(example["image"]).read_bytes()).hexdigest() != trial["image_sha256"]:
                raise ValueError(f"Image changed for {trial['example_id']}.")
            response_path = Path(trial["response"])
            if not response_path.is_file():
                raise ValueError(f"Missing trial response: {response_path}")
            response = json.loads(response_path.read_text())
            validate_response(response, request)
            entries.extend(response["examples"])
        combined = {"examples": entries}
        result = score(combined, public, private)
        prediction = {entry["example_id"]: set(entry["valid_interventions"]) for entry in entries}
        scenes = {entry["example_id"]: entry for entry in private["examples"]}
        pairs = []
        for family in ("kitchen", "workshop", "storage"):
            group = {entry["scene"]["variant"]: entry["example_id"]
                     for entry in private["examples"] if entry["scene"]["family"] == family}
            base, cosmetic, edited = group["base"], group["cosmetic"], group["blocked-near"]
            pairs.append({
                "family": family,
                "cosmetic_invariance": prediction[base] == prediction[cosmetic],
                "cosmetic_both_correct": all(
                    prediction[key] == set(scenes[key]["valid_interventions"])
                    for key in (base, cosmetic)),
                "relevant_edit_both_correct": all(
                    prediction[key] == set(scenes[key]["valid_interventions"])
                    for key in (base, edited)),
            })
        result["matched_pairs"] = pairs
        result["configured_model_id"] = model
        results[model] = result
        for name, data in (("responses.json", combined), ("scores.json", result)):
            path = output / model / name
            content = dump(data)
            if path.exists() and path.read_text() != content:
                raise ValueError(f"Refusing to replace changed evaluation output: {path}")
            path.write_text(content)
    summary = {
        "models": results,
        "trial_count": manifest["trial_count"],
        "fresh_context_per_scene": True,
        "replicates_per_model_scene": 1,
        "symbolic_ablation_run": False,
        "unattributed_development_run_included": False,
        "interpretation": "Exploratory, correlated pilot scenes; no population-level inference.",
    }
    path = output / "summary.json"
    if path.exists() and path.read_text() != dump(summary):
        raise ValueError("Refusing to replace a changed summary.")
    path.write_text(dump(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "collect"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/repair-pilot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/repair-pilot-named"))
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.dataset, args.output)
        print(f"Prepared {manifest['trial_count']} isolated trials.")
    else:
        summary = collect(args.dataset, args.output)
        for model, result in summary["models"].items():
            print(model, json.dumps(result["accuracy"]))


if __name__ == "__main__":
    main()
