from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
import joblib

dataset =load_dataset("sakthivinash/Language_Detection")
df = dataset['train'].to_pandas()
print(df.head())
print(df['Language'].value_counts())
# 'texte' = nom de la colonne contenant les phrases
vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1,3))  # caractères 1 à 3-gram
X = vectorizer.fit_transform(df['Text'])
y = df['Language']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")