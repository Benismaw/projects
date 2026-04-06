import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

while True:
    phrase = input("Tapez une phrase (ou 'exit' pour quitter) : ")
    if phrase.lower() == 'exit':
        break
    vect = vectorizer.transform([phrase])
    langue = model.predict(vect)
    print("Langue prédite :", langue[0])
