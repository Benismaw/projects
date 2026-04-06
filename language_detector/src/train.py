from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


import joblib


# ── Dataset ─────────────────────────────────────────────────────────
dataset = load_dataset("sakthivinash/Language_Detection")
df = dataset['train'].to_pandas()
print(f"Dataset : {df.shape[0]} exemples, {df['Language'].nunique()} langues")

# ── Vectorizer amélioré ──────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    analyzer='char_wb',      # char + word boundaries → plus robuste
    ngram_range=(2, 4),      # bigrammes à quadrigrammes
    max_features=50000,      # plus de features
    sublinear_tf=True        # normalisation log
)

X = vectorizer.fit_transform(df['Text'])
y = df['Language']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Modèle ───────────────────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, C=5, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nAccuracy : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(classification_report(y_test, y_pred))

# ── Sauvegarde ───────────────────────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\nModèle sauvegardé !")