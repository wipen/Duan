# Duan

### Healthcare System 1

**Fast decisions · Calibrated confidence · Edge deployment**

Duan is a **decision-native** model for healthcare. Rather than generating long text, it returns a  
typed decision, a probability distribution over the options, and a confidence value directly in a  
**single forward pass**.

When confidence is insufficient, Duan is designed to hand the case to a human expert, or to a more  
deliberative **System 2**.

---

## Contents

1. [Model base](#1-model-base)
2. [Deployment and usage](#2-deployment-and-usage)
3. [Training method](#3-training-method)
4. [Data](#4-data)
5. [Evaluation](#5-evaluation)
6. [Capability boundaries](#6-capability-boundaries)
7. [Citation](#7-citation)
8. [License](#8-license)

---

## 1. Model base

Duan is a **fine-tune of Laya**. Laya is an open-source **non-autoregressive *System 1* decision  
engine** that produces typed decisions over 100+ languages in a single forward pass.

> CONVAI INNOVATIONS. *Laya: fast, non-autoregressive System 1 decision engine*. Apache-2.0.  
> <https://github.com/convaiinnovations/laya>

Laya ships three official checkpoints; Duan is built on the **multilingual** one:

| Checkpoint                                                | Encoder                |   Params | Token budget (`max_len` / `head_max_len`) | Languages                |
| --------------------------------------------------------- | ---------------------- | -------: | ----------------------------------------- | ------------------------ |
| `convaiinnovations/laya`                                  | ModernBERT-large       |     421M | 512 / 192                                 | English                  |
| **`convaiinnovations/laya-multilingual`** (**Duan base**) | `jhu-clsp/mmBERT-base` | **322M** | **1024 / 256**                            | **100+**                 |
| `convaiinnovations/laya-typed-decisions`                  | ModernBERT-large       |     421M | 1024 / 256                                | typed-decision workflows |

### 1.1 Architecture

Laya is a **non-autoregressive** architectural family with the following properties:

- **Strictly proper scoring rules.** The output distribution is trained not only with cross-entropy  
  but against proper scoring rules, so the probability parameters **can be read as probabilities**.
- **Typed questions + a single scorer.** Every task is expressed as a question with a type —  
  `choice` / `score` / `noul` — and the three types **share the same scorer**; the only structural  
  difference is a 3-row type embedding (`type_emb = Embedding(3, d)`). There is **no  
  fixed-class-count classification head**; the decision head applies a masked softmax over each  
  sample's own option set.
- **The option set is part of the input.** The label space is rendered into the sequence at  
  inference time, so the model is not restricted to a fixed list of decisions and can adapt flexibly  
  to different decision scenarios.

| Question type | `criteria` shape                    | Rendered as                | Gold key space   |
| ------------- | ----------------------------------- | -------------------------- | ---------------- |
| `choice`      | `{label: description}`              | `"<label>: <description>"` | the labels       |
| `score`       | `[level 0 description, level 1, …]` | `"level k: …"`             | `"0" … "n-1"`    |
| `noul`        | `{false: …, true: …}`               | `"false: no, it does not"` | `false` / `true` |

### 1.2 Sequence template and token budget

```text
[CLS] <type> question: <instructions + rendered options> [SEP] [MASK] opt … [SEP] <state> [SEP]
```

The token budget is **not a free parameter** — it follows the checkpoint's own  
`rl_agent_config.json` (`1024` / `256` for this base). Measured on this corpus, the longest single  
example is ≈**366 tokens**.

### 1.3 Output

```json
{
  "decision": "review",
  "probabilities": { "routine": 0.08, "review": 0.84, "urgent": 0.08 },
  "confidence": 0.91,
  "abstain": false
}
```

The three question types return **different payload keys**. `choice` returns `choice` +  
`probabilities`;

`score` returns `score` (an expected level) + `probabilities`, with no `choice` key;  
`noul` returns only `noul` = P(true), with no `probabilities` key.

### 1.4 Download

The weights are released on two platforms (identical content); pick either:

**Hugging Face** — <https://huggingface.co/wipen/Duan>

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download wipen/Duan
```

or in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("wipen/Duan")
```

**ModelScope** — <https://www.modelscope.cn/models/wipenhan/Duan>

```bash
pip install -U modelscope
modelscope download --model wipenhan/Duan
```

or in Python:

```python
from modelscope import snapshot_download
snapshot_download("wipenhan/Duan")
```

---

## 2. Deployment and usage

### 2.1 Requirements

| | |
| --- | --- |
| Python | 3.10 or newer |
| Hardware | CPU only — no GPU needed |
| Memory | ~2 GB (322 M parameters, 643 MB of weights) |
| Network | first run downloads the weights (~643 MB), see [§1.4](#14-download) |

### 2.2 Agent deployment

Paste this one-line prompt into any coding agent that can run shell commands (Codex, WorkBuddy, Claude Code, Cursor, …) — it clones the repository, installs, starts and verifies everything:

```
Clone https://github.com/wipen/Duan and cd into it, then deploy Duan and start the local web demo: the weights come from https://huggingface.co/wipen/Duan (if huggingface.co is unreachable, download from https://www.modelscope.cn/models/wipenhan/Duan and point DUAN_MODEL at that local directory); run pip install -r requirements.txt, then python serve.py, then open http://127.0.0.1:8000 and confirm /health returns ok and one example runs.
```

The agent needs PyPI access (install) and, on first run, Hugging Face access (weights). The prompt above already names a ModelScope fallback; if HF is merely unreachable from the network, a mirror also works: `export HF_ENDPOINT=https://hf-mirror.com`.

When it finishes, open <http://127.0.0.1:8000> to try it. Manual steps: [§2.3](#23-manual-deployment).

### 2.3 Manual deployment

#### 2.3.1 Install

From the repository root:

```bash
pip install -r requirements.txt
```

The file declares only `laya[serve]`; pip resolves it and installs `torch`, `transformers`, `safetensors`, `huggingface_hub`, `fastapi` and `uvicorn` as well.

#### 2.3.2 Start the server

```bash
python serve.py                        # 127.0.0.1:8000
python serve.py --port 8080            # another port
python serve.py --model <id-or-path>   # another checkpoint (HF id or local dir)
```

| Env var | Meaning | Default |
| --- | --- | --- |
| `DUAN_MODEL` | HF repo id or local directory | `wipen/Duan` |
| `DUAN_HOST` / `DUAN_PORT` | bind address / port | `127.0.0.1` / `8000` |
| `HF_ENDPOINT` | Hugging Face mirror | — |

The first start downloads and loads the weights; on CPU this takes about 20–30 s. It is ready when the log shows `Uvicorn running on http://127.0.0.1:8000`.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | web demo (Playground) |
| `GET` | `/health` | health probe |
| `POST` | `/predict` | `{"state": …, "questions": …}` → decision |
| `GET` | `/docs` | interactive API reference |

#### 2.3.3 Try the demo

Open <http://127.0.0.1:8000> in a browser and, in the Playground:

1. pick one case under **Example use cases** (real cases from the test split);
2. click **Run**;
3. you should get the decision, a probability bar per option, and the confidence. Switch EN / 中文 at the top right.

Or call the API directly:

```bash
curl -s http://127.0.0.1:8000/predict -H 'Content-Type: application/json' -d '{
  "state": {"age": 67, "chief_complaint": "chest pain", "heart_rate": 118},
  "questions": {"acuity": {"type": "choice",
    "instructions": "What ESI acuity level should this visit be triaged to?",
    "criteria": {"1": "immediate life-saving", "2": "high risk", "3": "many resources",
                 "4": "one resource", "5": "no resources"}}}
}'
```

Expected: `answers.acuity.choice` (the ESI level), `answers.acuity.probabilities` (per-level bars) and `confidence`.

#### 2.3.4 Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'laya'` | dependencies missing → `pip install -r requirements.txt` |
| first start hangs on download | `huggingface.co` unreachable → `export HF_ENDPOINT=https://hf-mirror.com` |
| `Address already in use` | port taken → `python serve.py --port 8080` |
| page opens but **Run** errors | backend down or wrong address → start `python serve.py`; set the demo's **Server** box to the full URL, e.g. `http://127.0.0.1:8000/predict` |
| one decision takes ~0.4 s | normal on CPU; avoid firing many requests at once |

---

## 3. Training method

Training is a **full-parameter fine-tune** of the multilingual base, distributed with DDP, driven by  
Laya's **RLCD** objective.

### 3.1 The objective: RLCD

RLCD uses **strictly proper scoring rules** as the reward and a GRPO-style policy gradient.  
Concretely:

| Component                   |     Value | Role                                                                    |
| --------------------------- | --------: | ----------------------------------------------------------------------- |
| `group_size`                |         4 | GRPO baseline sampling group                                            |
| `sigma_start` → `sigma_end` | 0.4 → 0.1 | exploration noise, linearly annealed                                    |
| `w_sph`                     |      0.75 | spherical score weight                                                  |
| `w_rps`                     |       1.0 | ranked probability score weight — **applies to `score` questions only** |
| `ce_weight`                 |       1.0 | soft-label cross-entropy weight, added to the policy-gradient term      |

The reward is a proper scoring rule rather than "points for answering correctly", which means the  
model is optimised to recognise the **uncertainty** of its answer.

### 3.2 Optimization and batching

| Setting                                 | Value                                 |
| --------------------------------------- | ------------------------------------- |
| `epochs`                                | **8**                                 |
| `micro_batch`                           | 8                                     |
| `target_effective_batch`                | **64** sequences/step                 |
| `max_tokens_per_batch`                  | 4096                                  |
| `lr_encoder`                            | 2.5e-5                                |
| `lr_head`                               | 1.0e-4                                |
| `weight_decay` / `min_lr` / `grad_clip` | 0.01 / 1.0e-6 / 1.0                   |
| precision                               | bf16 (auto: compute capability ≥ 8.0) |
| gradient / head checkpointing           | on                                    |
| `freeze_encoder`                        | false — full fine-tune                |
| seed / calibration seed                 | 20260925 / 20260922                   |

### 3.3 Post-training temperature calibration

Temperature scaling happens **after** training, fitted on a held-out calibration slice, and it is  
fitted **per bucket** rather than as one global value. The bucket name is determined by  
`(question type, option count)`. Merging data with mixed option counts under one temperature  
concentrates the entire calibration error on whichever bucket is the minority.

### 3.4 Splits and anti-leakage

- The corpus is split into `train` and `test`.
- Leakage removal: **500 exact duplicates** between MedQA-Mainland's official train and test splits  
  were dropped from train (see [§4.4](#44-known-data-defects-we-corrected)).

---

## 4. Data

The corpus is **8 public sources across two languages (Chinese and English)**, converted uniformly  
into Laya's typed-decision format.

### 4.1 Sources

| # | Source                      | Language | Task                                                   | Options | train / test         | Label nature                 | License          |
| - | --------------------------- | -------- | ------------------------------------------------------ | ------- | -------------------- | ---------------------------- | ---------------- |
| 1 | NHAMCS ED                   | en       | ED triage (ESI 1–5)                                    | 5       | 20,000 / 2,000       | observed clinical outcome    | MIT              |
| 2 | UCI Diabetes                | en       | 30-day readmission window                              | 3       | 20,000 / 2,000       | observed clinical outcome    | UCI ML Repo      |
| 3 | MedQA (English)             | en       | medical knowledge single-choice                        | 5       | 11,450 / 1,145       | **exam answer key**          | see §4.3         |
| 4 | MedQA (Simplified Chinese)  | zh-Hans  | medical knowledge single-choice                        | 5       | 20,000 / 2,000       | **exam answer key**          | see §4.3         |
| 5 | MedQA (Traditional Chinese) | zh-Hant  | medical knowledge single-choice                        | 4       | 12,701 / 1,270       | **exam answer key**          | see §4.3         |
| 6 | CARE-Bench                  | en       | patient-facing triage and action recommendation        | 4       | 790 / 79             | annotated expert consensus   | **CC-BY-NC-4.0** |
| 7 | DDXPlus                     | en       | automatic medical diagnosis and differential diagnosis | 2–10    | 20,000 / 2,000       | **fully synthetic**          | CC-BY-4.0        |
| 8 | PubMedQA (PQA-L)            | en       | conclusion supported (yes/no)                          | 2       | 445 / 44             | literature expert annotation | MIT              |
|   | **Total**                   |          |                                                        |         | **105,386 / 10,538** |                              |                  |


> **Sources 3–5, 7 and 8 are not clinical outcome data.** MedQA labels are exam answer keys,  
> CARE-Bench labels are expert consensus grades, PubMedQA labels are literature annotations, and  
> DDXPlus labels come from a generative model. **None of them supports any claim about clinical  
> prediction, incidence or causality.** They are in the corpus so that the typed-decision machinery  
> is trained across multiple languages and option counts.

### 4.2 Shape of the corpus

| Quantity                    | Value                                                                                                    |
| --------------------------- | -------------------------------------------------------------------------------------------------------- |
| train / test rows           | **105,386 / 10,538** (test = 10.00%)                                                                     |
| train by language           | en **72,685** (69.0%) · zh-Hans **20,000** (19.0%) · zh-Hant **12,701** (12.0%)                          |
| balancing                   | per-source equal weight, capped at 20,000 rows per source                                                |
| option-count spread (train) | 2: 1,435 · 3: 21,597 · 4: 14,920 · 5: **53,007** · 6: 1,586 · 7: 1,288 · 8: 1,556 · 9: 1,208 · 10: 8,789 |
| corpus size on disk         | train 122.8 MB · test 12.3 MB (jsonl)                                                                    |

### 4.3 Licensing

Model weights are licensed under **CC-BY-NC-4.0, non-commercial use only**; the code is **Apache-2.0** (see [§8](#8-license)).

| Source         | Licence                    | What it means here                            |
| -------------- | -------------------------- | --------------------------------------------- |
| NHAMCS ED      | MIT                        | freely usable and redistributable             |
| UCI Diabetes   | UCI ML Repository terms    | check terms before redistribution             |
| PubMedQA       | MIT                        | freely usable                                 |
| DDXPlus        | CC-BY-4.0                  | commercial use permitted **with attribution** |
| **CARE-Bench** | **CC-BY-NC-4.0**           | **non-commercial only**                       |
| MedQA          | per upstream dataset terms | see the source distribution                   |

**This repository does not ship the training corpus.** The converted corpus is 135 MB of jsonl whose  
rows contain verbatim source text. The corpus is therefore always **regenerated locally from the  
original sources by the conversion scripts**, deterministically, with a fixed seed.

### 4.4 Known data defects we corrected

| Source         | Defect found                                                                                                                                          | Handling                                                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| MedQA Mainland | **genuine cross-split leakage: 221 test rows are exact duplicates of train rows** (6.45%); 32 rows in dev∩test; a further 910 duplicates within train | leakage removed from train (500 rows) plus 1,140 within-batch duplicates                                                                |
| MedQA Taiwan   | 9 rows overlap with the held-out set                                                                                                                  | dropped                                                                                                                                 |
| DDXPlus        | the correct pathology sits at **position 0** of the differential list **73.7%** of the time                                                           | options deterministically shuffled by content hash; first-position hit rate falls to **17.3%**, consistent with the uniform expectation |
| DDXPlus        | `icd10` and `severity` are **deterministic functions of the label**                                                                                   | excluded from the input state, kept only in metadata                                                                                    |
| PubMedQA       | ships its own baseline model predictions (`reasoning_free_pred`, `reasoning_required_pred`)                                                           | asserted out of the input state                                                                                                         |
| CARE-Bench     | `information_state` is 100% pure for one class; `round_role` reaches 84% purity                                                                       | removed from the input state, purity printed as evidence                                                                                |
| NHAMCS ED      | post-triage interventions, wait time, and vitals-missingness are all label proxies                                                                    | excluded from the input state                                                                                                           |

---

## 5. Evaluation

Metrics are chosen from the label distributions rather than from convention — accuracy alone is  
uninformative here: on NHAMCS triage the **majority class** already reaches **49.8%** on this test  
split.

The numbers below are measured on the `test` split (10,538 rows) **after an 8-epoch full fine-tune**.

| Task                            |     n | Accuracy | Majority baseline |   Macro-F1 |    ECE |
| ------------------------------- | ----: | -------: | ----------------: | ---------: | -----: |
| `ed_triage` (ESI 1–5)           | 2,000 |   0.5600 |            0.4975 | **0.3490** | 0.1423 |
| `readmission_window` (3 levels) | 2,000 |   0.5810 |            0.5350 |     0.4178 | 0.0391 |
| `ddx_diagnosis` (2–10 options)  | 2,000 |   0.9905 |            0.1660 |     0.9900 | 0.0071 |
| `medqa_mcq` (5 / 5 / 4 options) | 4,415 |   0.3647 |            0.2258 |     0.3659 | 0.4564 |
| `care_escalation` (4 levels)    |    79 |   0.6456 |            0.3418 |     0.6317 | 0.2951 |
| `pubmedqa` (yes/no)             |    44 |   0.7273 |            0.6364 |     0.6966 | 0.2727 |

---

## 6. Capability boundaries

1. **No causal inference.** Statistical-association and spontaneous-reporting sources cannot  
   support "drug X causes event Y", and no source of that class is used for any decision claim in  
   this project.
2. **No incidence estimation.**
3. **No clinical performance claim yet.** All measured numbers are listed in [§5](#5-evaluation),  
   with their denominators and limitations. The 0.50–0.65 range often cited for 5-class ESI  
   prediction is a literature expectation, **not a result of this project**.
4. **No substitute for clinical judgement.** Recall on high-risk minority classes is inherently  
   limited; cases of that kind should be routed to a human via a `confidence` threshold.
5. **No relicensing of third-party data.** See [§4.3](#43-licensing); CARE-Bench is non-commercial  
   and that constraint propagates to models trained on it.

---

## 7. Citation

```bibtex
@misc{duan2026,
  title        = {Duan: A Healthcare System 1 Decision Model},
  author       = {Han, Weipeng},
  year         = {2026},
  howpublished = {\url{https://github.com/wipen/Duan}},
  note         = {Work in progress}
}
```

If you use the underlying engine, cite Laya as well:

```bibtex
@misc{laya,
  title        = {Laya: fast, non-autoregressive System 1 decision engine},
  author       = {{ConvAI Innovations}},
  howpublished = {\url{https://github.com/convaiinnovations/laya}},
  note         = {Apache-2.0}
}
```

---

## 8. License

Code: **Apache-2.0** (see `LICENSE`); weights: **CC-BY-NC-4.0, non-commercial use only** (see `LICENSE-MODEL`).

Dataset licences are independent of this repository — see [§4.3](#43-licensing).

---

## Safety

Duan is intended for **research and clinical decision-support development**. It is not a substitute  
for qualified healthcare professionals, and must not be used for unvalidated autonomous diagnosis,  
prescribing, or other high-risk medical decisions.
