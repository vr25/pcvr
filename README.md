# PCVR: Preservation-Constrained Visual Repair: Evaluating Complete Sets of Scene Interventions

PCVR asks for every valid scene intervention, all minimum-cost repairs,
a no-solution verdict, and per-candidate preservation violations.
This package contains the experiment code and data used in the paper.

## Contents

| Path | Purpose |
|---|---|
| `repair_pilot.py` | Scene generation, rendering, geometric checker, response validation, scoring, and local-only reference policy |
| `named_repair_eval.py` | Prepare frontier-model trials and collect archived decisions |
| `local_vlm_eval.py` | Run and evaluate open-weight models through Ollama |
| `test_repair_pilot.py`, `test_named_repair_eval.py` | Dataset, checker, and frontier-evaluation tests |
| `requirements.txt` | Python dependency manifest |
| `artifacts/repair-pilot/` | Public requests, private oracle geometry/labels, 12 task images, overview image, and reference-policy report |
| `artifacts/repair-pilot-named/` | 24 frontier-model trial prompts, requests, decisions, manifest, and scores |
| `artifacts/repair-pilot-local/` | 144 open-model image trials and evaluation summary |
| `artifacts/repair-pilot-local-text/` | 144 open-model exact-coordinate trials and evaluation summary |

The earlier unidentified development evaluation, which is excluded from
the paper's reported comparisons, is not included. Cache files, local
handoff notes, manuscript sources, PDFs, and paper-build outputs are
also excluded.

## Dataset composition

| Setting | Retrieval goal | Scenes | Candidates | Valid / invalid |
|---|---|---:|---:|---:|
| Kitchen | Mug to countertop | 4 | 24 | 5 / 19 |
| Workshop | Drill to workbench | 4 | 24 | 5 / 19 |
| Storage | Parcel to packing table | 4 | 24 | 5 / 19 |
| Total | | 12 | 72 | 15 / 57 |

Each setting contains base, cosmetic, cheaper-placement-blocked, and
no-solution variants, with 2, 2, 1, and 0 valid candidates respectively.
Settings share a reflected canonical layout; the scenes are a controlled
diagnostic pilot rather than independently sampled environments.

The checker tests positive-area rectangle intersection and room
containment. Edge contact is allowed. Labels concern final-state
placements among the six listed candidates, not physical reachability,
transit paths, grasping, or dynamics.

## Setup

Use Python 3.10 or newer. The original environment used Python 3.12.2
and Pillow 10.4.0.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Run all commands below **from this package's root directory**.

## Reproduce scores without model calls

```bash
python3 -m unittest -q test_repair_pilot test_named_repair_eval
python3 named_repair_eval.py collect
python3 local_vlm_eval.py evaluate
python3 local_vlm_eval.py evaluate --output artifacts/repair-pilot-local-text
```

These evaluation commands rescore the bundled responses. They do not
contact model providers or require Ollama. They write evaluation
summaries in the corresponding artifact directories.

Frontier responses undergo strict consistency validation. Open-model
valid and cheapest sets are derived from their violation lists;
agreement with their explicitly stated sets is reported separately.
The two scoring conventions are intentionally different.

## Generate the scenes independently

Use a fresh destination rather than overwrite the archived dataset:

```bash
python3 repair_pilot.py generate --output artifacts/repair-pilot-regenerated
```

Rendering and oracle labels use fixed authored coordinates. Pillow
version differences may change image bytes even when geometry and
labels agree.

## Optional new model trials

New open-weight runs require a separately installed Ollama service and
downloaded models. The paper used Ollama 0.34.4 with default 4-bit
Q4_K_M builds of Qwen2.5-VL-3B, -7B, -32B, and Gemma-3-4B.
Each model/scene was run at temperature 0 with seed 0, and temperature
0.7 with seeds 1 and 2.

```bash
ollama pull qwen2.5vl:7b
# Start Ollama if it is not already running.
ollama serve
```

In another terminal, from this package's root:

```bash
python3 local_vlm_eval.py run --models qwen2.5vl:7b --repeats 3 \
  --output artifacts/new-local-image
python3 local_vlm_eval.py evaluate --output artifacts/new-local-image

python3 local_vlm_eval.py run --models qwen2.5vl:7b --repeats 3 \
  --text-only --output artifacts/new-local-text
python3 local_vlm_eval.py evaluate --output artifacts/new-local-text
```

For frontier trials, `named_repair_eval.py prepare --output
artifacts/new-frontier` prepares requests and instructions but does
not itself invoke models. The original trials used one fresh Copilot
agent context per scene and configured model alias
(`gpt-6.1-sol` or `claude-opus-5.5`). A compatible harness must execute
each prepared prompt and save its response before collection.
Provider-resolved versions, harness system prompts, and sampling
settings were not exposed. Blinding was instruction-based, not a
filesystem sandbox.

## Archive portability and provenance

For this distributable copy, machine-specific path prefixes in the
frontier manifest, requests, and prompt directory annotations were
replaced with package-root-relative paths. These are metadata-only
changes: model responses, scene images, image hashes, geometry, and
oracle labels are unchanged. Run collection from the package root so
those relative paths resolve correctly.

Gold labels are included for reproducibility. Do not expose
`examples.private.json`, reports, or other trial responses to a model
when conducting a new blinded evaluation.
