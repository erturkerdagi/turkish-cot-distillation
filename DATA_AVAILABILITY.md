# Data and model availability

Everything needed to verify the paper's numbers is openly available. Nothing is "available
on request".

## In this repository

| Item | Location |
|---|---|
| All code: teacher generation, simplification, training, evaluation, analysis | `scripts/` |
| All computed statistics (seeds, ANOVA, paired tests, bootstrap intervals, budget sweep, base models) | `results/revH_sonuclar.json` |
| Training-target statistics with the real tokenizer, truncation rates, matching audit, length correlations | `results/revI_denetim.json` |
| Output word counts per condition | `results/revJ_sozcuk.json` |
| Exploration and length-controlled analyses | `results/revTUMU_sonuclar.json` |
| Every table in the paper, as CSV | `results/tables/` |

The analysis scripts in `scripts/06_analysis/` run on these JSON files without a GPU, so
every statistic in the paper can be recomputed from this repository alone.

## Deposited separately

These exceed what Git should carry and are deposited outside the repository.

### Generated datasets

| File | Rows | Contents |
|---|---|---|
| `asama1_ogretmen_zincirleri.jsonl` | 1,000 | raw teacher chains |
| `asama2_sadelesmis.jsonl` | 525 | chains that passed both filters, with every simplified and baseline version of each target |
| `kesif_kayitlar.json` | 209 | Turkish–English morpheme and token comparison records |

### Trained adapters

46 LoRA adapters: 12 main runs (4 students × 3 data versions), 22 ablation and
length-matched-baseline runs, 12 additional seed runs. Base models are downloaded from the
Hugging Face Hub by the scripts; only the adapters are deposited.

### Evaluation outputs

74 JSONL files with per-item records — for every test problem: correctness, word count,
morpheme load, whether the budget was exhausted, output token count, how the answer was
extracted, the predicted and the true answer, and the generated text.

### Where to find them

- **Hugging Face Hub** — data and per-item evaluation outputs: https://huggingface.co/datasets/erturkerdagi/turkish-cot-distillation
- **Hugging Face Hub** — the 46 trained LoRA adapters: https://huggingface.co/erturkerdagi/turkish-cot-distillation-adapters
- **Zenodo** — archived snapshot of this repository, DOI [10.5281/zenodo.24429605](https://doi.org/10.5281/zenodo.24429605) (concept DOI; always resolves to the latest version)

The Zenodo DOI, not the GitHub URL, is the persistent identifier to cite and the one entered
in the journal's data availability statement: a GitHub repository can change, a Zenodo record
is frozen.

## Third-party sources

| Source | License |
|---|---|
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | MIT |
| [`malhajar/gsm8k-tr`](https://huggingface.co/datasets/malhajar/gsm8k-tr) | per dataset card |
| [`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) | MIT |
| Qwen2.5-0.5B / 1.5B / 3B | Apache 2.0 |
| Llama-3.2-3B | Llama 3.2 Community License |

`malhajar/gsm8k-tr` is a Turkish translation of GSM8K. The paper describes how it was
produced and what is and is not known about its translation quality.
