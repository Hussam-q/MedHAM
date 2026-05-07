```markdown
# MedHAM

**Evaluating Hallucination, Accuracy, and Misinformation in Large Language Models for Medical Question Answering: A Comparative Study of Retrieval-Augmented Generation and Citation Prompting**

MedHAM is a research codebase for evaluating large language model responses to evidence-based medical questions. It focuses on three outcome constructs: **hallucination**, **accuracy**, and **medical misinformation**.

This repository contains the source code used to create the SQLite database, collect model responses, run the multi-layer evaluation framework, analyse the results, and build the interactive dashboard.

- Dataset: [Hussam-q/MedHAM](https://huggingface.co/datasets/Hussam-q/MedHAM)
- Dashboard: [Hussam-q/MedHAM-Dashboard](https://huggingface.co/spaces/Hussam-q/MedHAM-Dashboard)

## Project Overview

The project evaluates whether retrieval-augmented generation and citation prompting improve the reliability of LLM answers to medical questions.

The released evaluation dataset contains 16,000 model responses and their associated evaluation results.

## Research Questions

| ID | Question |
|---|---|
| **RQ1** | To what extent do large language models produce hallucinated, inaccurate, and medically misinformative responses when answering evidence-based medical questions? |
| **RQ2** | How do retrieval-augmented generation and citation prompting, separately and together, affect hallucination, accuracy, and misinformation in large language model responses to medical questions, and do these effects differ across models and question difficulty levels? |

## Hypotheses

| ID | Hypothesis |
|---|---|
| **H2.1** | Retrieval-augmented generation will reduce hallucination and misinformation rates and improve accuracy scores relative to the no-retrieval baseline. |
| **H2.2** | The effect of retrieval augmentation and citation prompting on hallucination and misinformation rates and accuracy scores will be stronger for smaller models than for larger models. |
| **H2.3** | Citation prompting will reduce hallucination and misinformation rates and improve accuracy scores relative to zero-shot prompting. |
| **H2.4** | The combination of retrieval-augmented generation and citation prompting will yield lower hallucination and misinformation rates and higher accuracy scores than retrieval augmentation or citation prompting applied independently. |

## Repository Structure

```text
MedHAM/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── requirements.txt
├── src/
│   └── medham/
│       ├── schema.py
│       ├── pipeline.py
│       └── labeler.py
├── notebooks/
│   └── analysis.ipynb
└── dashboard/
    └── app.py
```

## Main Components

### `src/medham/schema.py`

Creates the local SQLite database, defines the relational schema, seeds static lookup tables, and loads the MedHallu source items.

### `src/medham/pipeline.py`

Collects LLM responses across all tested models and prompt strategies.

### `src/medham/labeler.py`

Runs the evaluation framework. It computes automated signals, verifies citations, performs claim-level factual checking, runs blind LLM judge evaluation, and writes final aggregated labels.

### `notebooks/analysis.ipynb`

Contains the statistical analysis, hypothesis testing, and figure generation.

### `dashboard/app.py`

Streamlit application for exploring the released MedHAM evaluation dataset.

## Experimental Design

The study uses a 2 x 2 factorial design crossing two intervention factors:

| Prompt format | No RAG | RAG |
|---|---|---|
| Zero-shot | `ZS_NoRAG` | `ZS_RAG` |
| Citation-required | `CIT_NoRAG` | `CIT_RAG` |

Each medical question is evaluated across four conditions and four tested models.

| Component | Count |
|---|---:|
| Medical questions | 1,000 |
| Tested models | 4 |
| Prompt strategies | 4 |
| Model responses | 16,000 |
| Blind judge evaluations | 64,000 |

## Tested Models

| Model | Provider / deployment |
|---|---|
| GPT-4o | OpenAI |
| Claude Sonnet 4.6 | Anthropic |
| Gemini 2.5 Flash | Google |
| Llama 3.1 8B | Meta model served via Groq |

## Judge Models

| Model | Provider / deployment |
|---|---|
| GPT-4o | OpenAI |
| Claude Sonnet 4.6 | Anthropic |
| Gemini 2.5 Flash | Google |
| DeepSeek V4 Flash | DeepSeek |

## Evaluation Framework

MedHAM uses a five-layer evaluation framework. The goal is not to rely on a single metric, but to combine complementary evidence sources.

| Layer | Method | Purpose |
|---|---|---|
| 1a | BioBERT semantic similarity | Measures semantic overlap between response and verified ground truth |
| 1b | MedNLI | Detects contradiction against ground truth or retrieved context |
| 2 | Citation verification | Checks DOI, PMID, and author-year citations using CrossRef and PubMed |
| 3 | PubMed-FActScore | Decomposes responses into atomic claims and checks support in PubMed |
| 4 | Blind LLM judge panel | Scores accuracy, hallucination, and misinformation |
| 5 | Rule-based aggregation | Produces final labels and consensus scores |

Final outcome labels are stored in `evaluation_results`.

## Dataset

The Hugging Face dataset contains eight exported tables produced from the normalized SQLite evaluation database.

| Table | Rows | Description |
|---|---:|---|
| `benchmark_items` | 1,000 | Source medical questions, verified ground truths, difficulty labels, and hallucination categories |
| `models` | 5 | Metadata for tested models and judge models |
| `prompt_strategies` | 4 | Prompting and RAG conditions |
| `rag_contexts` | 1,000 | knowledge context per question |
| `model_responses` | 16,000 | Raw generated LLM responses |
| `evaluation_signals` | 16,000 | Automated evaluation signals |
| `judge_evaluations` | 64,000 | Per-judge structured votes |
| `evaluation_results` | 16,000 | Final labels and consensus scores |

## Key Outcome Columns

The main outcome table is `evaluation_results`.

| Column | Description |
|---|---|
| `accuracy_score` | Final accuracy label: 0 = inaccurate, 1 = partly accurate, 2 = accurate |
| `hallucination_label` | Binary hallucination label |
| `misinformation_label` | Binary medical misinformation label |
| `accuracy_consensus` | Weighted mean of judge accuracy votes on a 0-2 scale |
| `hallucination_consensus` | Weighted fraction of judges voting hallucination present |
| `misinformation_consensus` | Weighted fraction of judges voting misinformation present |
| `track_consensus` | Whether automated signals and judge panel were directionally consistent |

The main automated-signal table is `evaluation_signals`.

| Column group | Examples |
|---|---|
| Semantic similarity | `biobert_precision`, `biobert_recall`, `biobert_f1` |
| NLI signals | `misinfo_nli_signal`, `intrinsic_nli_signal` |
| Citation verification | `citation_real_count`, `citation_fake_count`, `citation_verdict` |
| Claim support | `factscore_f1`, `factscore_supported`, `factscore_total` |
| Automated track verdict | `automated_track_verdict` |

## Loading the Dataset

Using `datasets`:

```python
from datasets import load_dataset

results = load_dataset("Hussam-q/MedHAM", "evaluation_results", split="train")
signals = load_dataset("Hussam-q/MedHAM", "evaluation_signals", split="train")
responses = load_dataset("Hussam-q/MedHAM", "model_responses", split="train")
judges = load_dataset("Hussam-q/MedHAM", "judge_evaluations", split="train")
```

Using pandas:

```python
import pandas as pd

results = pd.read_parquet(
    "hf://datasets/Hussam-q/MedHAM/evaluation_results.parquet"
)
```

## Setup

Clone the repository:

```bash
git clone https://github.com/Hussam-q/MedHAM.git
cd MedHAM
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local environment file:

```bash
cp .env.example .env
```

Then add the required API keys and contact emails to `.env`.

Required for `pipeline.py`:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
GROQ_API_KEY
```

Required for `labeler.py`:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
DEEPSEEK_API_KEY
PUBMED_EMAIL
CROSSREF_MAILTO
```

Optional:

```text
NCBI_API_KEY
HF_TOKEN
```

## Running the Pipeline

Run the scripts in this order:

```bash
python src/medham/schema.py
python src/medham/pipeline.py
python src/medham/labeler.py
```

The scripts generate a local SQLite database named:

```text
src/medham/medham.db
```

This file is generated locally and ignored by Git.

## Running the Dashboard Locally

```bash
streamlit run dashboard/app.py
```

The dashboard loads the released MedHAM dataset from Hugging Face and provides interactive views for:

- overall hallucination, misinformation, and accuracy rates
- model-level comparisons
- RAG and citation-prompting treatment effects
- hypothesis test summaries
- response-level data exploration

## Source Data

The source medical questions, ground truths, and knowledge contexts are derived from:

[UTAustin-AIHealth/MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu)

Specifically, this project uses the `pqa_labeled` split, which contains human-verified medical question-answering examples.

## Reproducibility Notes

Key settings used in the response-generation and evaluation pipeline:

| Parameter | Value |
|---|---|
| Generation temperature | 0.0 |
| Maximum response tokens | 600 |
| Judge temperature | 0.0 |
| PubMed-FActScore threshold | 0.85 |
| Judge aggregation threshold | 0.50 |

Model providers may update hosted model behavior over time. For interpretation of the released results, use the stored dataset tables and response timestamps rather than assuming future API calls will reproduce identical text.

## Medical Safety Notice

This project is for research and evaluation purposes only. It is not a clinical decision-support system and should not be used to guide diagnosis, treatment, or patient care.

## License

The code in this repository is released under the MIT License.

The released dataset is available on Hugging Face under the license specified on the dataset card.

## Author

Hussam Alqahtani
