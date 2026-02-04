# Repair-of-Thought (ROT)

Repair-of-Thought is a function-level automated program repair pipeline for Java bugs.  
Given buggy code and failing test context, ROT:

1. generates reasoning-based fix suggestions (`thinker`),
2. synthesizes full-function patches (`action`),
3. evaluates patch quality (`eval`) as `CORRECT`, `PLAUSIBLE`, or `INCORRECT`.

The benchmark used in this repository is `datasets/defects4j-sf.json` (522 single-function Defects4J bugs).

## Repository At A Glance

- `src/thinker/` - reasoning and suggestion generation
- `src/action/` - patch synthesis from suggestions
- `src/eval/` - semantic evaluation and patch labeling
- `datasets/` - benchmark inputs
- `outputs/` - generated solutions, patches, and evaluation files

## Reproducing The Study

### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set required environment variables:

```bash
export PYTHONPATH=src
export TOGETHER_API_KEY=...
export SELF_API_URL=...
export SELF_API_ID=...
export SELF_API_TOKEN=...
export GEMINI_API_KEY=...
mkdir -p outputs/sol outputs/patches outputs/val
```

### 2. Single-Bug Smoke Test

```bash
BUG=Math-2

venv/bin/python -m thinker.thinker \
  -d datasets/defects4j-sf.json \
  -o outputs/sol/${BUG}.json \
  -eo outputs/sol/${BUG}_extracted.json \
  -s 1 \
  -bug ${BUG} \
  -patch_num 3

venv/bin/python -m action.action \
  -d datasets/defects4j-sf.json \
  -s outputs/sol/${BUG}_extracted.json \
  -o outputs/patches/${BUG}_patch.json \
  -bug ${BUG}

venv/bin/python -c "import asyncio,os; from eval.auto_eval import evaluate_patches; asyncio.run(evaluate_patches('${BUG}', f'outputs/patches/${BUG}_patch.json', os.environ['GEMINI_API_KEY']))"
```

Expected files:
- `outputs/sol/${BUG}.json`
- `outputs/sol/${BUG}_extracted.json`
- `outputs/patches/${BUG}_patch.json`
- `outputs/val/${BUG}_patch_val.json`

### 3. Full Benchmark Run (522 Bugs)

Run the same three stages for every bug in `datasets/defects4j-sf.json` with fixed settings:
- `sample_size = 1`
- `patch_num = 3`

Recommended execution policy:
- stage-wise per bug (`thinker -> action -> eval`)
- deterministic environment (fixed package versions)
- periodic checkpointing of `outputs/` for recovery

### 4. Aggregate Outcomes

Each bug-level evaluation is written to `outputs/val/*_patch_val.json`.  
You can aggregate final counts by best-per-bug status (`CORRECT > PLAUSIBLE > INCORRECT`) using your analysis script or notebook.

## Notes On Evaluation

- `CORRECT`: semantically equivalent to benchmark fix
- `PLAUSIBLE`: likely fixes trigger behavior but not equivalent
- `INCORRECT`: fails semantic checks

The evaluator combines AST-based, symbolic-style structural checks, and LLM-based assessment (see `src/eval/auto_eval.py`).

## 👥 Contributors

- **Satwik Pandey** - [GitHub](https://github.com/satwik-pandey)
- **Suresh Raghu** - [Github](https://github.com/R-Suresh07)