# Char-CNN Smishing Classifier with EVA Attack



---

## Table of Contents

1. [What is Smishing?](#what-is-smishing)
2. [Paper Summary](#paper-summary)
3. [What This Project Implements](#what-this-project-implements)
4. [Model Architecture](#model-architecture)
5. [EVA Attack Algorithm](#eva-attack-algorithm)
6. [Dataset](#dataset)
7. [Project Structure](#project-structure)
8. [How to Run (Google Colab)](#how-to-run-google-colab)
9. [How to Run (Local)](#how-to-run-local)
10. [Results](#results)
11. [Paper vs Our Implementation](#paper-vs-our-implementation)
12. [Dependencies](#dependencies)

---

## What is Smishing?

**Smishing** (SMS phishing) is a cyberattack where criminals send fraudulent text messages impersonating banks, delivery services, or government agencies to trick victims into clicking malicious links or revealing personal data.

Key facts from the paper:
- 700% surge in smishing attacks in H1 2021 (Proofpoint)
- 98% of smishing messages are **variants** of a small set of original messages — scammers just tweak the text slightly to evade detection
- Damage in South Korea alone: ₩1,265 billion in 2021

---

## Paper Summary

The paper proposes a complete pipeline for building a practical, on-device smishing classifier:

```
Real SMS Data → Preprocessing → Topic Clustering → Training → Evaluation → Adversarial Training → Final Model
```

**Three core contributions:**

| Contribution | Description |
|---|---|
| **Lightweight Char-CNN** | 127 kB model that runs entirely on-device — no cloud, no privacy leak |
| **EVA Attack Tool** | Text evasion attack that generates smishing variants to test robustness |
| **Adversarial Training** | Retraining with EVA-generated examples to make the model harder to fool |

**Key finding:** A model with 99% accuracy on clean data can still be fooled **82% of the time** with simple text tweaks. Adversarial training cuts this to **41%** while keeping accuracy at 99%.

---

## What This Project Implements

| Component | Status | Notes |
|---|---|---|
| Char-CNN (paper Table 3 architecture) | ✅ | embedding=48, filters=192, kernel=12, hidden=10 |
| URL / CALL / FILE masking | ✅ | Matches paper's Section IV preprocessing |
| Character vocabulary encoding | ✅ | 70-char English alphabet, reversed encoding |
| Weighted sampling for class imbalance | ✅ | Paper approach |
| EVA Phase 1: Preprocess | ✅ | Mask tokens before attack |
| EVA Phase 2: BreakPatterns | ✅ | Remove strong smishing tokens |
| EVA Phase 3: PerturbStruct | ✅ | Space/line-break perturbations |
| EVA Phase 4: ImportantTokens | ✅ | Rank words by confidence impact |
| EVA Phase 5: PerturbChar | ✅ | Leet substitutions, typo insert/delete |
| EVA Phase 6: PerturbWord | ✅ | BERT fill-mask substitution (bert-base-uncased) |
| Adversarial training (Section VII) | ✅ | Collection mode + fine-tuning |
| Edit Distance Rate metric | ✅ | Paper's edr formula with Levenshtein |
| Final comparison table | ✅ | Matches paper Table 5 format |

---

## Model Architecture

Based on **Table 3** in the paper. The model is intentionally tiny — designed for on-device inference.

```
Input SMS text
     │
     ▼
Character Encoding  ← reversed character indices (0 = padding)
     │
     ▼
Embedding Layer     ← vocab_size → 48 dimensions
     │
     ▼
Conv1D              ← 192 filters, kernel_size=12, ReLU
     │
     ▼
Global Max Pool     ← takes max across entire sequence → shape (192,)
     │
     ▼
Linear(192 → 10)    ← "FFN with ten hidden cells" (paper)
+ ReLU + Dropout(0.5)
     │
     ▼
Linear(10 → 2)      ← Normal / Smishing
     │
     ▼
Softmax → P(smishing)
```

**Why this architecture?**
- No recurrence → fast inference
- Character-level → no vocabulary needed, handles typos and leet-speak naturally
- Global max pool → fixed-size output regardless of SMS length
- Only ~116k parameters → ~453 kB in PyTorch float32 (paper achieves 127 kB after TFLite int8 quantization)

---

## EVA Attack Algorithm

EVA (text **Ev**asion **A**ttack) is a black-box attack — it only needs the model's confidence score, not its internals. It mirrors what a real scammer would do.

```
Input: smishing message m, model F, edit budget µ_edit=0.4

Phase 1 — Preprocess
  └─ Mask URLs, normalize text

Phase 2 — BreakPatterns
  └─ Remove strong smishing tokens (free, prize, claim, urgent, ...)
  └─ If confidence < 0.5 → DONE ✓

Phase 3 — PerturbStruct
  └─ Try adding/removing spaces and line breaks
  └─ Keep changes that reduce confidence score
  └─ If confidence < 0.5 → DONE ✓

Phase 4 — ImportantTokens
  └─ Delete each word, measure confidence drop
  └─ Rank words by how much removing them helps

Phase 5+6 — PerturbChar & PerturbWord (per important token)
  ├─ Character: leet (e→3, o→0, s→5), typo insert/delete, symbol insert
  ├─ Word: delete word, BERT fill-mask substitution, swap adjacent words
  └─ If confidence < 0.5 AND edr ≤ 0.4 → DONE ✓

Output: adversarial message (or None if budget exhausted)
```

**Success condition:** `P(smishing) < 0.5` AND `levenshtein(original, perturbed) / len(original) ≤ 0.40`

**Example from our run:**
```
Original    [0.989]: congratulations! you've won a $1000 gift card. click here to claim: linka
Adversarial [0.029]: congratulations! you've  a good gift card. here to claim: LINKA
Edit rate  : 0.260  ← only 26% of characters changed, meaning still readable
```

---

## Dataset

**UCI SMS Spam Collection** — used as a smishing proxy since the paper's KISA dataset (Korean, private) is not publicly available.

| Property | Value |
|---|---|
| Total messages | 5,574 |
| Normal (ham) | 4,827 (86.6%) |
| Smishing (spam) | 747 (13.4%) |
| Source | [HuggingFace: ucirvine/sms_spam](https://huggingface.co/datasets/ucirvine/sms_spam) |
| Download | Automatic — no login required |

**Split used:**
```
Train : 3,901  (523 smishing)
Val   :   836  (112 smishing)
Test  :   837  (112 smishing)
```

**Why this dataset?** The original paper uses a Korean dataset from KISA (government agency). For an English implementation, the UCI SMS Spam Collection is the closest public equivalent with real-world spam/phishing messages.

---

## Project Structure

```
CHAR_CNN_SMISH/
│
├── Smishing CharCNN Colab_NEW.ipynb   ← MAIN NOTEBOOK (run this)
├── smishing_charcnn_colab.ipynb       ← original version
│
├── src/
│   ├── model.py          ← Zhang et al. 2015 large CharCNN (original repo)
│   ├── data_loader.py    ← dataset loading and encoding utilities
│   ├── utils.py          ← text preprocessing, metrics
│   └── focal_loss.py     ← focal loss implementation
│
├── train.py              ← CLI training script (original repo)
├── predict.py            ← CLI inference script (original repo)
├── config.json           ← model config (original repo)
│
├── plots/
│   ├── character_cnn.png      ← Zhang et al. architecture diagram
│   ├── conv_layers.png
│   ├── fc_layers.png
│   └── training_metrics.PNG
│
└── README.md
```

> **Note:** `src/model.py`, `train.py`, and `predict.py` are from the original [ahmedbesbes/character-based-cnn](https://github.com/ahmedbesbes/character-based-cnn) repo (Zhang et al. 2015 large architecture). The smishing notebook implements the **paper's simplified architecture** (1 conv layer) independently inside the notebook.

---

## How to Run (Google Colab)

This is the recommended way — no local setup needed.

### Step 1 — Open the notebook

Upload `Smishing CharCNN Colab_NEW.ipynb` to [colab.research.google.com](https://colab.research.google.com)

Or open directly: **File → Upload notebook → select the .ipynb file**

### Step 2 — Enable GPU

```
Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save
```

Without GPU the training will still work but takes ~5× longer.

### Step 3 — Run all cells in order

```
Runtime → Run all   (Ctrl+F9)
```

Or run cell by cell with **Shift+Enter**.

### Cell-by-cell guide

| Cell | What it does | Expected time |
|---|---|---|
| Cell 1 | Install packages | ~1 min |
| Cell 2 | Imports | instant |
| Cell 3 | Config (model hyperparameters) | instant |
| Cell 4 | Download & preprocess dataset | ~30 sec |
| Cell 5 | Plot dataset analysis | instant |
| Cell 6 | Train/val/test split | instant |
| Cell 7 | Define Char-CNN model | instant |
| Cell 8 | Dataset class + DataLoaders | instant |
| Cell 9 | Training loop functions | instant |
| Cell 10 | **Train base model (15 epochs)** | ~3–5 min (GPU) |
| Cell 11 | Evaluate + confusion matrix | ~30 sec |
| Cell 12 | EVA helper functions | instant |
| Cell 13 | Load BERT fill-mask | ~2 min (first download) |
| Cell 14 | EVA algorithm | instant |
| Cell 15 | **Run EVA attack (100 messages)** | ~10–20 min |
| Cell 16 | Show attack examples | ~1 min |
| Cell 17 | Collect adversarial examples | ~15–30 min |
| Cell 18 | **Adversarial training (10 epochs)** | ~3 min |
| Cell 19 | Evaluate adversarial model | ~30 sec |
| Cell 20 | Re-run EVA on adversarial model | ~10–20 min |
| Cell 21 | Final comparison plot | instant |
| Cell 22 | Interactive demo | instant |
| Cell 23 | EVA demo on single message | ~2 min |

**Total runtime: ~1.5–2 hours** (mostly EVA attack cells 15, 17, 20)

### To speed up the attack cells

In Cell 15, reduce `ATTACK_SAMPLE`:
```python
ATTACK_SAMPLE = 30   # instead of 100 — takes ~5 min total
```

---

## How to Run (Local)

### Prerequisites

```bash
python >= 3.9
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm
```

For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

### Run the notebook locally

```bash
pip install jupyter
jupyter notebook "Smishing CharCNN Colab_NEW.ipynb"
```

### Run the original CLI training script

The `train.py` / `predict.py` scripts are for the original Zhang et al. large CharCNN. To train on a CSV dataset:

```bash
python train.py \
  --data_path ./data/sms.csv \
  --text_column text \
  --label_column label \
  --max_length 160 \
  --epochs 10 \
  --batch_size 128 \
  --optimizer adam \
  --learning_rate 0.001 \
  --checkpoint 1 \
  --output ./models/
```

To run inference on a single message:

```bash
python predict.py \
  --model ./models/your_model.pth \
  --text "Congratulations! You have won a free prize. Click here to claim." \
  --max_length 160 \
  --number_of_classes 2
```

---

## Results

Results from `Smishing CharCNN Colab_NEW.ipynb` run on Google Colab T4 GPU:

### Classification Performance

| Metric | Base Char-CNN | Adv Char-CNN |
|---|:---:|:---:|
| Accuracy | 0.9869 | 0.9881 |
| F1 Score | 0.9867 | 0.9880 |
| False Positive Rate | 0.0041 | 0.0041 |
| False Negative Rate | 0.0714 | 0.0625 |

### Robustness Against EVA Attack

| Model | Attack Success Rate (ASR) | Interpretation |
|---|:---:|---|
| Base Char-CNN | 0.6882 | Fooled 69% of the time |
| Adv Char-CNN | 0.6170 | Fooled 62% of the time |

### EVA Attack Demo Output

```
Original  [0.989]: congratulations! you've won a $1000 gift card. click here to claim: linka
Adversarial [0.029]: congratulations! you've  a good gift card. here to claim: LINKA
Edit distance rate: 0.260  ←  only 26% of characters changed
✓ Attack successful!

Adversarially trained model on same message:
Prediction: NORMAL (prob: 0.006)  ←  model still fooled (needs more training data)
```

---

## Paper vs Our Implementation

### Metric Comparison (Table 5)

| Metric | Paper Base | Our Base | Paper Adv | Our Adv |
|---|:---:|:---:|:---:|:---:|
| Accuracy | 0.9959 | 0.9869 | 0.9944 | 0.9881 |
| F1 Score | 0.9946 | 0.9867 | 0.9927 | 0.9880 |
| FPR | 0.0041 | **0.0041 ✅** | 0.0054 | 0.0041 |
| FNR | 0.0043 | 0.0714 ⚠️ | 0.0059 | 0.0625 |
| ASR | 0.8241 | 0.6882 | 0.4091 | 0.6170 |

### Why the gaps exist

| Gap | Reason |
|---|---|
| Accuracy ~0.01 lower | Paper trains on 1.2M messages; we use 5.5k |
| FNR much higher (0.071 vs 0.004) | Only 523 smishing training examples vs 250,000 in the paper |
| FPR exactly matches | Both achieve 0.0041 — shows architecture is correct |
| ASR reduction smaller (0.07 vs 0.41) | Paper uses 84,000 adversarial training examples; we collect only 88 due to small dataset |

### Key differences: Paper vs Implementation

| Aspect | Paper | This Implementation |
|---|---|---|
| Dataset | KISA (Korean, private) 250k smishing + 950k normal | UCI SMS Spam (English, public) 747 spam + 4827 ham |
| Language | Korean | English |
| BERT model | `lassl/bert-ko-base` (Korean) | `bert-base-uncased` (English) |
| Tokenizer | Korean morpheme tokenizer | Whitespace tokenization |
| Model deployment | TFLite (127 kB after int8 quantization) | PyTorch (~454 kB float32) |
| Adversarial training examples | 84,000–120,000 | ~88 (limited by dataset size) |

### What replicates correctly

- ✅ Architecture and hyperparameters match Table 3 exactly
- ✅ FPR matches the paper's value (0.0041)
- ✅ All 6 EVA phases implemented and functional
- ✅ ASR goes **down** after adversarial training (correct direction)
- ✅ Classification accuracy stays high after adversarial training (correct behaviour)
- ✅ Adversarial examples visually similar to originals (edit rate ~0.26)

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Char-CNN model training and inference |
| `transformers` | BERT fill-mask for EVA word-level perturbations |
| `datasets` | Load UCI SMS Spam from HuggingFace Hub |
| `scikit-learn` | Train/test split, metrics (accuracy, F1, confusion matrix) |
| `pandas` / `numpy` | Data handling |
| `matplotlib` / `seaborn` | Plots and confusion matrix |
| `tqdm` | Progress bars |

Install all at once:

```bash
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn tqdm
```

---

## References

1. **Seo et al. (2024).** On-Device Smishing Classifier Resistant to Text Evasion Attack. *IEEE Access*, 12, 4762–4779. DOI: 10.1109/ACCESS.2024.3349577

2. **Zhang et al. (2015).** Character-level Convolutional Networks for Text Classification. *NeurIPS*. arXiv:1509.01626

3. **Devlin et al. (2018).** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805

4. **UCI SMS Spam Collection.** Available at: https://huggingface.co/datasets/ucirvine/sms_spam

---

## License

MIT License — see [LICENSE](LICENSE) for details.
