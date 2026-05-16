# Zero-Shot Bias Detection in LLMs: A Multi-Model Analysis on BBQ

Systematic evaluation of bias in small language models using zero-shot NLI (Natural Language Inference) on the BBQ benchmark dataset.

## Project Overview

This research project automates bias detection across 10+ small language models by performing zero-shot inference on the BBQ (Bias Benchmark for QA) dataset. Each model acts as a zero-shot classifier using NLI to predict bias in question-answer pairs across 11 demographic categories.

**Key contributions:**
- Systematic bias profiling of small NLI models
- Multi-category analysis (age, gender, race, religion, SES, disability, etc.)
- Batch inference with timestamped results for reproducibility
- Aggregated bias metrics and cross-model comparisons

## Dataset Structure

The BBQ dataset is split into 11 bias categories:

| Category | File |
|----------|------|
| Age | `BBQ-AGE.json` |
| Disability | `BBQ-DISABILITY.json` |
| Gender | `BBQ-GENDER.json` |
| Nationality | `BBQ-NATIONALITY.json` |
| Physical Appearance | `BBQ-PHYSICAL_APPEARANCE.json` |
| Race/Ethnicity | `BBQ-RACE_ETHNICITY.json` |
| Race + Gender | `BBQ-RACE_GENDER.json` |
| Race + SES | `BBQ-RACE_SES.json` |
| Religion | `BBQ_RELIGION.json` |
| Socioeconomic Status (SES) | `BBQ_SES.json` |
| Sexual Orientation | `BBQ_SEXUAL_ORIENTATION.json` |

Additional metadata: `additional_metadata.csv`, `Data_date.txt`

## Models Evaluated

All models are zero-shot NLI classifiers (no fine-tuning on BBQ):

- `facebook/bart-large-mnli`
- `roberta-large-mnli`
- `valhalla/distilbart-mnli-12-3`
- `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
- `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- `Jiva/xlm-roberta-large-it-mnli`
- `NbAiLab/nb-bert-base-mnli`
- `Recognai/zeroshot_selectra_small`
- `cross-encoder/nli-MiniLM2-L6-H768`
- `cross-encoder/nli-deberta-base`

## Directory Descriptions

- **`data/`** - Contains raw BBQ (Bias Benchmark for QA) datasets and data processing scripts
- **`models/`** - Inference Script
- **`results/`** - Main results directory with analysis utilities, aggregated metrics, and per-model outputs
  - **`file_tools/`** - Utility scripts for result analysis and processing
  - **`metrics/`** - CSV files containing aggregated bias metrics across models and categories
  - **`[model_name]/`** - Individual model result directories (10 models total), each containing parquet files with raw predictions