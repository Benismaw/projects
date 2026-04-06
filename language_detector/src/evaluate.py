from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Charger modèle + données ────────────────────────────────────────
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

dataset = load_dataset("sakthivinash/Language_Detection")
df = dataset['train'].to_pandas()

from sklearn.model_selection import train_test_split
_, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Language'])

X_test = vectorizer.transform(df_test['Text'])
y_test = df_test['Language']
y_pred = model.predict(X_test)

# ── Métriques ────────────────────────────────────────────────────────
print(f"Accuracy globale : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"Nombre de langues : {df['Language'].nunique()}")
print(f"Nombre d'exemples test : {len(df_test)}")

# ── Matrice de confusion ─────────────────────────────────────────────
langs = sorted(df['Language'].unique())
cm = confusion_matrix(y_test, y_pred, labels=langs)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=langs, 
            yticklabels=langs, cmap='Blues')
plt.title("Matrice de confusion — Language Detector (98.2%)")
plt.ylabel("Vraie langue")
plt.xlabel("Langue prédite")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("../results/confusion_matrix.png", dpi=150)
plt.show()

# ── Langues les plus difficiles ──────────────────────────────────────
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).T.iloc[:-3]
df_report = df_report.sort_values('f1-score')
print("\nLangues les plus difficiles :")
print(df_report.head(5)[['precision', 'recall', 'f1-score']])
