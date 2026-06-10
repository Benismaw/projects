# 🌍 Automatic Language Detector

A supervised machine learning system that automatically detects the language of a text sample among **30 languages**, achieving **98.2% accuracy**.

## 🎯 Results

| Metric | Score |
|--------|-------|
| Accuracy | **98.2%** |
| Languages supported | **30** |
| Dataset size | ~32,000 samples |

## 🧠 Approach

The system uses a classic NLP pipeline:

1. **Preprocessing** — text cleaning and normalization
2. **Feature extraction** — TF-IDF vectorization on character n-grams
3. **Classification** — Logistic Regression

### Why character n-grams?

Character n-grams capture language-specific patterns (accents, character combinations) that are highly discriminative across languages — even for languages without spaces like Chinese or Japanese.

### Why Logistic Regression over Naive Bayes?

I initially tried Naive Bayes (80% accuracy), but it assumes feature independence — which is false with TF-IDF. Logistic Regression learns optimal weights per feature without this constraint, jumping accuracy to **98.2%**.

## 🗂️ Supported Languages

Arabic, Chinese, Dutch, English, Estonian, French, German, Greek, Hindi, Indonesian, Japanese, Korean, Latin, Persian, Portugese, Pushto, Romanian, Russian, Spanish, Swedish, Tamil, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and more.

## 🛠️ Tech Stack

- Python 3.x
- scikit-learn
- NumPy
- Pandas

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Benismaw/projects.git
cd projects/language_detector

# Install dependencies
pip install -r requirements.txt

# Run the detector
python predict.py --text "Bonjour, comment allez-vous ?"
```

## 📊 How It Works

```
Input text
    ↓
TF-IDF vectorization (character n-grams)
    ↓
Logistic Regression classifier
    ↓
Predicted language + confidence score
```

## 📁 Project Structure

```
language_detector/
├── train.py          # Training script
├── predict.py        # Inference script
├── model/            # Saved model weights
├── data/             # Dataset
└── requirements.txt
```

## 💡 Key Learnings

- Character n-grams are more robust than word n-grams for language detection
- Stratified train/test split is essential to ensure all 30 languages are represented
- Model comparison (Naive Bayes vs Logistic Regression) showed a **+18% accuracy improvement**

## 📄 License

MIT
