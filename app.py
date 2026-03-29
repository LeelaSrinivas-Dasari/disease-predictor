import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Disease Predictor", page_icon="🏥", layout="wide")
import gdown

url = "https://drive.google.com/file/d/1v9P_W1BGWI4zlQXIFuUYutpnWeTkLi4m/view?usp=sharing"
output = "model.pkl"

gdown.download(url, output, quiet=False)
# ── Load model ───────────────────────────────────────────────
bundle   = joblib.load('model.pkl')
model    = bundle['model']
le       = bundle['le']
SYMPTOMS = bundle['symptoms']
desc_df  = bundle['description']
prec_df  = bundle['precaution']
sev_df   = bundle['severity']

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

SYMPTOM_OPTIONS = [s.replace('_', ' ').title() for s in SYMPTOMS]

# ── Helper functions ─────────────────────────────────────────
def get_vector_from_text(text):
    t = text.lower()
    for phrase, sym in sorted(SYNONYMS.items(), key=lambda x: -len(x[0])):
        t = t.replace(phrase, sym.replace('_', ' '))
    vector = []
    for s in SYMPTOMS:
        sym_words = s.replace('_', ' ')
        matched = any(word in t for word in sym_words.split() if len(word) > 3)
        vector.append(1 if matched else 0)
    return vector

def get_vector_from_selected(selected):
    selected_raw = [s.lower().replace(' ', '_') for s in selected]
    return [1 if s in selected_raw else 0 for s in SYMPTOMS]

def merge_vectors(v1, v2):
    return [max(a, b) for a, b in zip(v1, v2)]

def get_description(disease):
    row = desc_df[desc_df['Disease'] == disease]
    return row.iloc[0]['Description'] if not row.empty else "No description available."

def get_precautions(disease):
    row = prec_df[prec_df['Disease'] == disease]
    if row.empty:
        return []
    return [row.iloc[0][c] for c in ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
            if pd.notna(row.iloc[0][c])]

def get_severity(detected):
    total = 0
    for s in detected:
        sym = s.lower().replace(' ', '_')
        row = sev_df[sev_df['Symptom'] == sym]
        if not row.empty:
            total += int(row.iloc[0]['weight'])
    return total

# ── UI ───────────────────────────────────────────────────────
st.title("🏥 Disease Predictor")
st.caption("Select symptoms from the dropdown or type them below. You can use both.")

st.divider()

selected_symptoms = st.multiselect(
    "Select Symptoms",
    options=SYMPTOM_OPTIONS,
    placeholder="Search and select symptoms..."
)

text_input = st.text_area(
    "Or type your symptoms",
    placeholder="e.g. I have high fever, vomiting and headache"
)

predict_btn = st.button("Predict Disease", type="primary", use_container_width=True)

st.divider()

# ── Prediction ───────────────────────────────────────────────
if predict_btn:
    if not selected_symptoms and not text_input.strip():
        st.warning("Please select or type at least one symptom.")
    else:
        v1 = get_vector_from_selected(selected_symptoms)
        v2 = get_vector_from_text(text_input) if text_input.strip() else [0] * len(SYMPTOMS)
        vector = merge_vectors(v1, v2)

        if sum(vector) == 0:
            st.warning("No symptoms matched. Try selecting from the dropdown.")
        else:
            X = pd.DataFrame([vector], columns=SYMPTOMS)

            disease  = le.inverse_transform(model.predict(X))[0]
            proba    = model.predict_proba(X)[0]
            conf     = round(max(proba) * 100, 1)
            detected = [s.replace('_', ' ').title() for s, v in zip(SYMPTOMS, vector) if v == 1]

            top3_idx = proba.argsort()[::-1][:3]
            top3     = [(le.inverse_transform([i])[0], round(proba[i]*100, 1)) for i in top3_idx]

            # SHAP
            pred_class = int(model.predict(X)[0])
            shap_out   = bundle['explainer'].shap_values(X)
            shap_arr   = np.array(shap_out)
            if shap_arr.ndim == 3:
                sv = shap_arr[pred_class % shap_arr.shape[0], 0, :]
            elif shap_arr.ndim == 2:
                sv = shap_arr[0, :]
            else:
                sv = shap_arr.flatten()
            sv = sv.flatten().tolist()

            top_shap  = sorted(range(len(sv)), key=lambda i: -abs(sv[i]))[:5]
            shap_data = [(SYMPTOMS[i].replace('_', ' ').title(), round(sv[i], 3))
                         for i in top_shap if vector[i] == 1]

            severity_score = get_severity(detected)

            # ── Results ──────────────────────────────────────
            st.subheader("Results")

            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Disease", disease)
            col2.metric("Confidence", f"{conf}%")
            if severity_score <= 5:
                col3.metric("Severity", "🟢 Mild", f"Score: {severity_score}")
            elif severity_score <= 10:
                col3.metric("Severity", "🟡 Moderate", f"Score: {severity_score}")
            else:
                col3.metric("Severity", "🔴 Severe", f"Score: {severity_score}")

            st.divider()

            # Top 3
            st.subheader("Top 3 Predictions")
            for name, prob in top3:
                st.progress(prob / 100, text=f"{name} — {prob}%")

            st.divider()

            # Detected symptoms
            st.subheader("Symptoms Detected")
            st.write(", ".join(detected) if detected else "None")

            st.divider()

            # Description + Precautions
            col4, col5 = st.columns(2)
            with col4:
                st.subheader("About this Disease")
                st.write(get_description(disease))

            with col5:
                st.subheader("Precautions")
                precs = get_precautions(disease)
                if precs:
                    for i, p in enumerate(precs, 1):
                        st.write(f"{i}. {p}")
                else:
                    st.write("No precautions found.")

            st.divider()

            # SHAP
            if shap_data:
                st.subheader("Key Symptoms (SHAP)")
                df_shap = pd.DataFrame(shap_data, columns=["Symptom", "SHAP Score"])
                st.dataframe(df_shap, use_container_width=True)
