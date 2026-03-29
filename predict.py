import joblib
import pandas as pd

bundle   = joblib.load('model.pkl')
model    = bundle['model']
le       = bundle['le']
SYMPTOMS = bundle['symptoms']

SYNONYMS = {
    'high fever': 'high_fever', 'high temperature': 'high_fever',
    'tired': 'fatigue', 'exhausted': 'fatigue',
    'throwing up': 'vomiting', 'feel sick': 'nausea',
    'stomach pain': 'stomach_pain', 'body ache': 'muscle_pain',
    'sore throat': 'throat_irritation', 'migraine': 'headache',
    'breathless': 'breathlessness', 'shortness of breath': 'breathlessness',
    'no appetite': 'loss_of_appetite', 'loose motion': 'diarrhoea',
    'diarrhea': 'diarrhoea', 'itchy': 'itching',
    'yellow skin': 'yellowish_skin', 'yellow eyes': 'yellowing_of_eyes',
    'blurry vision': 'blurred_and_distorted_vision',
    'dark urine': 'dark_urine', 'dizzy': 'dizziness',
    'runny nose': 'runny_nose', 'red eyes': 'redness_of_eyes',
}

def get_vector(text):
    t = text.lower()
    for phrase, sym in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        t = t.replace(phrase, sym.replace('_', ' '))
    return [1 if s.replace('_', ' ') in t else 0 for s in SYMPTOMS]

def predict(text):
    vector = get_vector(text)
    X = pd.DataFrame([vector], columns=SYMPTOMS)

    disease = le.inverse_transform(model.predict(X))[0]
    proba   = model.predict_proba(X)[0]
    conf    = round(max(proba) * 100, 1)

    print(f"Disease    : {disease}")
    print(f"Confidence : {conf}%")

def explain(text):
    vector = get_vector(text)
    X = pd.DataFrame([vector], columns=SYMPTOMS)

    pred_class = model.predict(X)[0]
    sv = bundle['explainer'].shap_values(X)[pred_class][0]

    print("Top symptoms driving prediction:")
    for i in sorted(range(len(sv)), key=lambda i: -abs(sv[i]))[:5]:
        if vector[i]:
            print(f"  {SYMPTOMS[i].replace('_', ' ')} — {sv[i]:.3f}")

# Test
predict("I have high fever, vomiting and headache")
explain("I have high fever, vomiting and headache")
