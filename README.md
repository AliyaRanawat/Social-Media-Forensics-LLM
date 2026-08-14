### Recommended Repository Names

Choose one of these clean, standard GitHub names:

* `adaptive-cyberbullying-detection` *(Recommended)*
* `social-media-forensics-llm`
* `genz-toxicity-detector`
* `cyberbullying-forensics-ensemble`

---

### Ready-to-Use `README.md`

Copy and paste the entire block below directly into your repository's `README.md` file:

```markdown
# Social Media Forensics: Adaptive Cyberbullying Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Interface-Gradio%20UI-orange)](https://gradio.app/)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.61%25-brightgreen.svg)]()

An end-to-end Machine Learning pipeline designed for **Social Media Forensics** to detect both **explicit toxicity** and nuanced **implicit/passive-aggressive Gen-Z cyberbullying** using Large Language Model (LLM) Sentence Transformers and a Soft-Voting Ensemble Classifier.

---

## 📌 Key Highlights
* **Massive 1-Lakh (100,000) Dataset:** Merges 95,000 real-world samples from the Google/Jigsaw Toxic Comment dataset with 5,000 custom oversampled Gen-Z sarcastic roasts, backhanded compliments, and wholesome slang.
* **Semantic Context Understanding:** Replaces traditional bag-of-words and TF-IDF with `all-MiniLM-L12-v2` Sentence Transformer embeddings extracted via GPU acceleration.
* **Tri-Model Soft Voting Ensemble:** Combines three diverse classifiers (`HistGradientBoosting`, `ExtraTrees`, and `MLPClassifier`) for robust decision boundaries.
* **96.61% Validated Accuracy:** High precision on both conventional toxic comments and modern internet vernacular.
* **Live Interactive Demo:** Includes a real-time Gradio Web Interface for instant comment classification.

---

## 🏗️ System Architecture

```text
[ Dataset A: Jigsaw 95k ]       [ Dataset B: Gen-Z Custom 5k ]
           \                         /
            \                       /
             ▼                     ▼
======================================================
                PRE-PROCESSING MODULE
     (Merge, Label Map, Shuffle 1-Lakh Samples)
======================================================
                          |
                          ▼
======================================================
                    LLM EXTRACTOR
       (all-MiniLM-L12-v2 extracts Embeddings)
======================================================
                          |
                          ▼
======================================================
              SUPREME VOTING CLASSIFIER
      (HistGradientBoosting + ExtraTrees + MLP)
======================================================
               /                     \
              ▼                       ▼
    [ 96.61% Accuracy ]      [ Live Web Interface ]

```

---

## 📊 Dataset Overview

1. **Google / Jigsaw Toxicity Dataset (95,000 Samples):** Balanced split (47,500 Toxic / 47,500 Safe) capturing explicit toxicity, threats, insults, and hate speech.
2. **Gen-Z & Implicit Toxicity Dataset (5,000 Samples):** Curated few-shot dataset augmented and oversampled to catch subtle passive-aggression (e.g., *"Bro thought he cooked 💀"*, *"You're actually pretty smart for someone who looks like you"*, *"You ate and left no crumbs ✨"*).

---

## ⚙️ Model Pipeline

| Stage | Technology / Component | Details |
| --- | --- | --- |
| **Embedding Extractor** | `sentence-transformers/all-MiniLM-L12-v2` | GPU batch encoding with L2 normalization |
| **Model 1** | `HistGradientBoostingClassifier` | Fast, histogram-based gradient boosted trees |
| **Model 2** | `ExtraTreesClassifier` | Extremely randomized decision trees (100 estimators) |
| **Model 3** | `MLPClassifier` | Multi-Layer Perceptron (256, 128 hidden layers) with early stopping |
| **Ensemble** | `VotingClassifier` | Soft-voting strategy based on predicted probabilities |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR-USERNAME/adaptive-cyberbullying-detection.git](https://github.com/YOUR-USERNAME/adaptive-cyberbullying-detection.git)
cd adaptive-cyberbullying-detection

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

*(Or install manually:)*

```bash
pip install numpy pandas torch scikit-learn sentence-transformers datasets gradio

```

### 3. Run the Pipeline & Launch UI

```bash
python main.py

```

---

## 📈 Performance & Results

* **Validated Test Accuracy:** **`96.61%`**
* **Train / Test Split:** 80% (80,000 samples) / 20% (20,000 samples)
* **Inference Speed:** Real-time (< 50ms per comment on GPU)

---

## 👩‍💻 Author

* **Aliya Ranawat**
*Department of Artificial Intelligence and Machine Learning*

```

```
