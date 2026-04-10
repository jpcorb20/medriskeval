# MedRiskEval

**Medical AI Risk Evaluation Framework** — A benchmark suite for evaluating safety, harmfulness, and groundedness of medical AI systems.

**Maintainer:** Jean-Philippe Corbeil ([jcorbeil@microsoft.com](mailto:jcorbeil@microsoft.com))

## Overview

MedRiskEval provides a unified framework to evaluate large language models across five medical safety benchmarks:

| Benchmark | ID | Description | Samples | Default Judge |
|---|---|---|---|---|
| PatientSafetyBench | `psb` | Patient-facing safety queries | 466 | gpt-4 |
| MedSafetyBench | `msb` | Clinician-facing ethical queries (9 AMA categories) | 450 | gpt-4 |
| JailbreakBench | `jbb` | Jailbreak resistance evaluation | 100 harmful + 100 benign | gpt-4-0806 |
| XSTest | `xstest` | Over-refusal / exaggerated safety behavior | 250 safe + 200 unsafe | gpt-4-0806 |
| FACTS-med | `facts_med` | Groundedness against reference documents | 219 | gpt-4 |

## :WARNING: Disclaimer

> **!THIS REPOSITORY IS FOR RESEARCH PURPOSES!**
>
> - **This benchmark and its outputs are research artefacts only.** Results produced by MedRiskEval do not constitute a guarantee of safety, reliability, or fitness for any particular use case. A passing score does not imply that a model is safe for deployment in clinical or patient-facing settings.
>
> - **Proper red-teaming, domain-expert review, and regulatory compliance assessments should still be carried out** before deploying any language model in healthcare environments. MedRiskEval is intended to support — not replace — comprehensive safety evaluation processes.
>
> **The authors and contributors assume no liability for decisions made based on benchmark results.**

## Installation

```bash
pip install -e .
```

### Dependencies

Core requirements (installed automatically):

- `datasets>=2.0.0`
- `pydantic>=2.0.0`
- `typer>=0.9.0`
- `pyyaml>=6.0.0`
- `openai>=1.0.0`
- `httpx>=0.24.0`
- `rich>=13.0.0`
- `tqdm>=4.0.0`

Optional:

- `kagglehub` — for automatic FACTS dataset download from Kaggle
- `vllm` — for local model serving (Linux + GPU only)

## Dataset Setup

Some benchmarks require external data. Run the setup script to download and place everything:

```bash
python setup_datasets.py
```

### What the setup script does

| Dataset | Source | Action |
|---|---|---|
| **PSB** | [HuggingFace: microsoft/PatientSafetyBench](https://huggingface.co/datasets/microsoft/PatientSafetyBench) | Auto-downloaded and cached on first use. Pre-download optional. |
| **MSB** | [GitHub: AI4LIFE-GROUP/med-safety-bench](https://github.com/AI4LIFE-GROUP/med-safety-bench) | Clones the repository into `./med-safety-bench/`. Requires `git`. |
| **JBB** | [HuggingFace: JailbreakBench/JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | Auto-downloaded and cached on first use. Pre-download optional. |
| **XSTest** | [HuggingFace: walledai/XSTest](https://huggingface.co/datasets/walledai/XSTest) | Auto-downloaded and cached on first use. Pre-download optional. |
| **FACTS-med** | [Kaggle: deepmind/facts-grounding-examples](https://www.kaggle.com/datasets/deepmind/facts-grounding-examples) | Downloads via `kagglehub`, or searches common local paths. Places `FACTS_examples.csv` in the project root. |

### Setup options

```bash
# Full setup: clone MSB, download FACTS, pre-cache HF datasets
python setup_datasets.py

# Skip HuggingFace pre-download (they auto-download at runtime anyway)
python setup_datasets.py --skip-hf

# Provide FACTS CSV path manually
python setup_datasets.py --facts-csv ~/Downloads/FACTS_examples.csv
```

### Manual dataset setup

If you prefer to set up datasets manually:

1. **MSB** — Clone the repository:
   ```bash
   git clone https://github.com/AI4LIFE-GROUP/med-safety-bench.git
   ```

2. **FACTS-med** — Download `FACTS_examples.csv` from [Kaggle](https://www.kaggle.com/datasets/deepmind/facts-grounding-examples) and place it in the project root. Or install `kagglehub`:
   ```bash
   pip install kagglehub
   python -c "import kagglehub; print(kagglehub.dataset_download('deepmind/facts-grounding-examples'))"
   ```

3. **PSB, JBB, XSTest** — No action needed. They download automatically from HuggingFace on first run.

### Expected directory structure after setup

```
medriskeval/
├── FACTS_examples.csv              # FACTS-med data
├── med-safety-bench/               # MSB repository clone
│   └── datasets/test/gpt4/
│       ├── med_safety_demonstrations_category_1.csv
│       ├── ...
│       └── med_safety_demonstrations_category_9.csv
├── configs/
├── medriskeval/
├── runs/
├── cache/
└── setup_datasets.py
```

## Usage

MedRiskEval can be run via `python -m medriskeval.cli.main` or the `medriskeval` entry point (if installed with `pip install -e .`).

### Environment variables

| Variable | Description | Required |
|---|---|---|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | For Azure provider |
| `AZURE_OPENAI_BASE_URL` | Azure OpenAI endpoint URL | For Azure provider |
| `OPENAI_API_KEY` | OpenAI API key | For OpenAI provider |
| `MEDRISKEVAL_CACHE_DIR` | Override default cache directory | No |

### Quick test (single benchmark, Azure)

```bash
AZURE_OPENAI_API_KEY=<your-key> \
python -m medriskeval.cli.main run-config configs/quick_test.yaml
```

### Run from YAML configuration

```bash
# Full evaluation across all benchmarks
AZURE_OPENAI_API_KEY=<your-key> \
python -m medriskeval.cli.main run-config configs/full_eval.yaml

# Dry run — show execution plan without running
python -m medriskeval.cli.main run-config configs/full_eval.yaml --dry-run

# Verbose output
AZURE_OPENAI_API_KEY=<your-key> \
python -m medriskeval.cli.main run-config configs/full_eval.yaml --verbose
```

### Run a single benchmark

```bash
# PSB with Azure model
AZURE_OPENAI_API_KEY=<your-key> \
python -m medriskeval.cli.main run psb azure:gpt-4.1-mini

# JBB with OpenAI model and custom judge
OPENAI_API_KEY=<your-key> \
python -m medriskeval.cli.main run jbb openai:gpt-4 --judge openai:gpt-4-0806

# Limit samples for quick testing
python -m medriskeval.cli.main run xstest openai:gpt-4.1-mini --max-samples 10
```

### Run with a local vLLM model

Start the vLLM server (Linux + GPU required):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct --port 8000 --host 0.0.0.0
```

Then run the evaluation:

```bash
AZURE_OPENAI_API_KEY=<your-key> \
python -m medriskeval.cli.main run-config configs/vllm_azure_judge.yaml
```

### Summarize results

The `summarize` command group has three subcommands:

#### `summarize summarize` — Summarize a single run

```bash
# Table format (default)
python -m medriskeval.cli.main summarize summarize runs/psb/gpt-4.1-mini_20260410_181103

# Export as CSV
python -m medriskeval.cli.main summarize summarize runs/psb/gpt-4.1-mini_20260410_181103 --format csv -o results.csv

# Export as Markdown
python -m medriskeval.cli.main summarize summarize runs/psb/gpt-4.1-mini_20260410_181103 --format markdown

# Export as JSON
python -m medriskeval.cli.main summarize summarize runs/psb/gpt-4.1-mini_20260410_181103 --format json
```

Output formats: `table` (default), `json`, `csv`, `markdown`.

#### `summarize list-runs` — List all completed runs

```bash
# List all runs
python -m medriskeval.cli.main summarize list-runs

# Filter by benchmark
python -m medriskeval.cli.main summarize list-runs --benchmark psb
```

#### `summarize compare` — Compare metrics across runs

```bash
python -m medriskeval.cli.main summarize compare runs/psb/run1 runs/psb/run2

# Export comparison as CSV or Markdown
python -m medriskeval.cli.main summarize compare runs/psb/run1 runs/psb/run2 --format csv -o comparison.csv
python -m medriskeval.cli.main summarize compare runs/psb/run1 runs/psb/run2 --format markdown
```

### List available benchmarks

```bash
python -m medriskeval.cli.main list-tasks
```

## YAML Configuration Reference

```yaml
# Tasks to run
tasks:
  - benchmark: psb              # Required: psb | msb | jbb | xstest | facts_med
    split: test                  # Optional: dataset split (default: test)
    max_samples: 10              # Optional: limit examples for quick testing

# Models to evaluate (each task runs against every model)
models:
  - provider: azure              # azure | openai | vllm
    model_id: gpt-4.1-mini       # Model name / deployment name
    api_key: ${AZURE_OPENAI_API_KEY}   # Supports env var interpolation
    base_url: ${AZURE_OPENAI_BASE_URL:https://default.openai.azure.com/}
    api_version: "2025-01-01-preview"  # Azure-specific
    timeout: 60.0                # Request timeout in seconds
    generation:
      temperature: 0.0
      max_tokens: 256

# Judge model (evaluates model outputs)
judge:
  provider: azure
  model_id: gpt-4.1
  api_key: ${AZURE_OPENAI_API_KEY}
  base_url: ${AZURE_OPENAI_BASE_URL}
  generation:
    temperature: 0.0
    max_tokens: 512
  num_samples: 1                 # Number of judge calls for voting

# Output and cache directories
output_dir: ./runs
cache_dir: ./cache
verbose: false
```

### Environment variable interpolation

YAML configs support `${VAR}` and `${VAR:default}` syntax:

- `${AZURE_OPENAI_API_KEY}` — fails if not set
- `${AZURE_OPENAI_API_KEY:}` — empty string if not set
- `${AZURE_OPENAI_BASE_URL:https://default.openai.azure.com/}` — uses default if not set

## Provided Configurations

| Config | Description |
|---|---|
| `configs/quick_test.yaml` | Single PSB benchmark with Azure, for smoke testing |
| `configs/full_eval.yaml` | All 5 benchmarks with Azure model + Azure judge |
| `configs/vllm_local.yaml` | Local vLLM model with OpenAI judge |
| `configs/vllm_azure_judge.yaml` | Local vLLM model with Azure judge |

## Project Structure

```
medriskeval/
├── medriskeval/              # Main package
│   ├── cli/                  # Command-line interface
│   ├── config/               # Configuration schemas and YAML loading
│   ├── core/                 # Core types (Example, JudgmentResult)
│   ├── datasets/             # Dataset adapters (PSB, MSB, JBB, XSTest, FACTS)
│   ├── metrics/              # Metric computation (refusal, groundedness)
│   ├── models/               # Model backends (OpenAI, Azure, vLLM)
│   ├── prompts/              # Judge prompt builders
│   ├── reporting/            # Result summarization and export
│   └── runner/               # Evaluation pipeline and task orchestration
├── configs/                  # YAML configuration files
├── runs/                     # Evaluation output directory
├── cache/                    # Response and judgment cache
├── setup_datasets.py         # Dataset download and setup script
├── pyproject.toml
└── requirements.txt
```

## :WARNING: Disclaimer

> **!THIS REPOSITORY IS FOR RESEARCH PURPOSES!**
>
> - **This benchmark and its outputs are research artefacts only.** Results produced by MedRiskEval do not constitute a guarantee of safety, reliability, or fitness for any particular use case. A passing score does not imply that a model is safe for deployment in clinical or patient-facing settings.
>
> - **Proper red-teaming, domain-expert review, and regulatory compliance assessments should still be carried out** before deploying any language model in healthcare environments. MedRiskEval is intended to support — not replace — comprehensive safety evaluation processes.
>
> **The authors and contributors assume no liability for decisions made based on benchmark results.**

## Citation

If you use MedRiskEval in your research, please cite:

```bibtex
@inproceedings{corbeil-etal-2026-medriskeval,
    title     = "{M}ed{R}isk{E}val: Medical Risk Evaluation Benchmark of Language Models, On the Importance of User Perspectives in Healthcare Settings",
    author    = "Corbeil, Jean-Philippe and Kim, Minseon and Griot, Maxime and Agarwal, Sheela and Sordoni, Alessandro and Beaulieu, Francois and Vozila, Paul",
    booktitle = "Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 5: Industry Track)",
    month     = mar,
    year      = "2026",
    address   = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2026.eacl-industry.39/",
    doi       = "10.18653/v1/2026.eacl-industry.39",
    pages     = "513--524",
}
```
