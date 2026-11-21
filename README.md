# Target Length Rather Than Morphological Weighting Reduces Overthinking in Turkish Chain-of-Thought Distillation

Code, analysis scripts and result files for a study on chain-of-thought (CoT) distillation
into small language models for Turkish, an agglutinative language.

Submitted to *Knowledge-Based Systems* (Ms. Ref. No. KNOSYS-D-26-20702).

---

## What this study found

The study began from the hypothesis that **morpheme load** — a language-sensitive cost
metric that counts morphemes rather than words — identifies which sentences of a teacher's
reasoning chain can be dropped without losing information, and that distilling students on
morpheme-simplified chains would reduce *overthinking* (students exhausting their token
budget before reaching an answer).

Simplification does reduce token exhaustion substantially: by **16.0 to 23.3 percentage
points** across the three Qwen students at a 512-token budget, with no detectable accuracy
cost. That part held up.

**The morphological part did not.** Controlled, length-matched ablations show:

| Evidence | Result |
|---|---|
| Uniform truncation (keep every *n*-th sentence; carries no linguistic information, and produces targets **22% longer** in tokens) | Matches the proposed method: 14.0% vs 15.0% exhaustion on Qwen2.5-1.5B, 19.0% vs 18.3% on Qwen2.5-3B |
| Sentence-count-matched random deletion | Slightly worse (22.3%, 21.0%) — so *which* sentences are dropped is not wholly immaterial |
| Morpheme vs word criterion | Identical keep/discard decision on **94.2%** of 6,123 sentences |
| Exhaustion vs training-target token length (8 conditions) | Pearson **r = 0.898** (p = 0.002) and **r = 0.947** (p = 0.0004) |
| "Answer is in the last sentence" as an alternative explanation | Ruled out: r = 0.08 (p = 0.85) and r = 0.05 (p = 0.91) |
| Morphological density predicting failure, after controlling for chain length | Not significant (p = 0.45) |
| Budget sweep | The advantage is confined to tight budgets and disappears at 1,024 tokens |

**Conclusion:** what the intervention actually manipulates is the **token length of the
training target**. The weighting criterion decides which sentences are removed but has no
measurable effect on the outcome, and morphological weighting confers no advantage over word
counting or over a fixed stride. Morpheme load remains useful as a *language-sensitive cost
metric* for agglutinative languages — it is not a causal driver of overthinking.

The original hypothesis is reported as falsified rather than quietly reframed. The ablations
that falsified it are in this repository.

---

## Experimental scope

- **Teacher:** `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- **Students:** `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Qwen2.5-3B`, `Llama-3.2-3B` (QLoRA, 4-bit NF4, bf16 compute; LoRA r=16, α=32, dropout=0.05)
- **Data:** GSM8K / [`malhajar/gsm8k-tr`](https://huggingface.co/datasets/malhajar/gsm8k-tr)
- **Pipeline:** 1,000 problems sent to the teacher → 525 chains passed both filters (teacher-correct **and** Turkish-consistent) → simplification → distillation
- **Evaluation:** 300 held-out test problems, deterministic generation, 512-token budget (plus a 256/512/1024/2048 sweep on 100 problems)
- **Totals:** 8 training conditions, 4 students, 3 random seeds, 4 token budgets — 46 adapters and 74 evaluation runs
- **Compute:** ~16 A100 GPU hours end to end

---

## Repository layout

```
scripts/
  01_exploration/        Turkish–English morpheme and token comparison
  02_teacher/            teacher chain generation and filtering
  03_simplification/     morpheme- and word-weighted sentence selection
  04_training/           QLoRA distillation (main runs + ablation runs)
  05_evaluation/         batched evaluation, seed runs, budget sweep
  06_analysis/           statistics, audits, figures
results/
  revH_sonuclar.json     main statistics: seeds, ANOVA, paired tests, budget sweep, base models
  revI_denetim.json      training-target statistics with the real tokenizer, matching audit, correlations
  revJ_sozcuk.json       output word counts per condition
  revTUMU_sonuclar.json  exploration and length-controlled analyses
  revDUZELTME_sonuclari.json  reference-target repair log
  tables/                every table in the paper, as CSV
```

Script names and in-code comments are in Turkish (the language of the study and of the data).
Output column names and the CSV tables are in English.

### Which script produces which part of the paper

| Script | Produces |
|---|---|
| `01_exploration/kesif_boru_hatti.py` | Table 1 — Turkish/English token and morpheme comparison |
| `02_teacher/asama1_ogretmen_uret.py` | teacher chains (`asama1_ogretmen_zincirleri.jsonl`) |
| `03_simplification/asama2_sadelestirme.py` | simplified versions (`asama2_sadelesmis.jsonl`) |
| `04_training/asama3_tam_egitim.py` | 12 main adapters |
| `04_training/rev_E_ablasyon_egitim.py` | 22 ablation and length-matched-baseline adapters |
| `05_evaluation/asama4_degerlendirme.py` | main evaluation, 300 problems |
| `05_evaluation/asama5_coklu_tohum.py` | seed runs |
| `05_evaluation/rev_F_degerlendirme.py` | batched evaluation with per-item records; budget sweep; base models |
| `06_analysis/rev_TUMU.py` | Sections 3.1.1, 5.2 — length-controlled analyses, rule-level ablation |
| `06_analysis/rev_H_final_analiz.py` | Tables 4, 6, 7, 8; Figures 8–9; ANOVA, McNemar, bootstrap, TOST |
| `06_analysis/rev_I_denetim.py` | Table 5b; truncation rates; matching audit; length correlations |
| `06_analysis/rev_J_sozcuk.py` | Table 3 word column |
| `06_analysis/rev_ENVANTER.py` | inventory check — confirms every expected adapter and evaluation file exists |

---

## Reproducing

### Requirements

GPU work was run on Colab with a single A100 (40 GB). The analysis scripts are CPU-only.

```bash
pip install -r requirements.txt
```

Turkish morphological analysis uses [Zeyrek](https://github.com/obulat/zeyrek); English
segmentation bounds use [Morfessor](https://github.com/aalto-speech/morfessor) plus a
rule-based affix counter (the two give an upper and a lower bound, reported as a range
rather than a point estimate).

### Setting the data directory

Every script reads its working directory from the `PROJE_DIZINI` environment variable:

```bash
export PROJE_DIZINI=/path/to/project
```

It defaults to `/content/drive/MyDrive/turkish-cot-distillation/Project` for Colab. The
directory holds the generated chains, the adapters and the evaluation outputs; the scripts
create the subfolders they need and never overwrite an existing run (they resume instead).

### Order

```
01_exploration  →  02_teacher  →  03_simplification  →  04_training  →  05_evaluation  →  06_analysis
```

The analysis scripts in `06_analysis/` can be run on the published `results/*.json` without
a GPU if you only want to check the statistics.

---

## Data and model availability

The code and all analysis outputs are in this repository. The generated datasets and the
trained LoRA adapters are too large for Git and are deposited separately — see
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

Nothing in this study requires access "on request": every number in the paper can be
recomputed from the deposited artifacts with the scripts here.

---

## Citation

If you use this code or these results, please cite the paper. See
[`CITATION.cff`](CITATION.cff); the entry will be updated with volume and DOI on publication.

---

## License

Code is released under the MIT License ([`LICENSE`](LICENSE)).

Derived datasets follow the license of their sources: GSM8K (MIT) and `malhajar/gsm8k-tr`.
Model adapters follow the licenses of their base models — Qwen2.5 (Apache 2.0) and
Llama 3.2 (Llama 3.2 Community License).
