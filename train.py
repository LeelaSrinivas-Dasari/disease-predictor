import pandas as pd
import joblib
import shap
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

import os
os.chdir(r'C:\Users\DELL\Desktop\LIV_PROJ\Disease_Pred\files')

SYMPTOMS = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
    'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue','muscle_wasting',
    'vomiting','burning_micturition','spotting_urination','fatigue','weight_gain',
    'anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness',
    'lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever',
    'sunken_eyes','breathlessness','sweating','dehydration','indigestion','headache',
    'yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes',
    'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
    'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
    'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
    'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion',
    'chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements',
    'pain_in_anal_region','bloody_stool','irritation_in_anus','neck_pain','dizziness',
    'cramps','bruising','obesity','swollen_legs','swollen_blood_vessels',
    'puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech',
    'knee_pain','hip_joint_pain','weakness_of_one_body_side','loss_of_smell',
    'bladder_discomfort','foul_smell_of_urine','continuous_feel_of_urine','passage_of_gases',
    'internal_itching','toxic_look_(typhos)','depression','irritability','muscle_pain',
    'altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation',
    'dischromic_patches','watering_from_eyes','increased_appetite','polyuria','family_history',
    'mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
    'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
    'distention_of_abdomen','history_of_alcohol_consumption','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples',
    'blackheads','scurring','skin_peeling','silver_like_dusting','small_dents_in_nails',
    'inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

# Load main dataset
df = pd.read_csv('disease_dataset.csv.csv')
df.columns = df.columns.str.strip()
df['Disease'] = df['Disease'].str.strip()
scols = [c for c in df.columns if c != 'Disease']

# Build binary matrix
rows = []
for _, row in df.iterrows():
    present = set(str(s).strip() for s in row[scols] if str(s).strip() not in ['nan', ''])
    r = {s: int(s in present) for s in SYMPTOMS}
    r['Disease'] = row['Disease']
    rows.append(r)

data = pd.DataFrame(rows)
X = data[SYMPTOMS]
y = data['Disease']

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

# Train voting classifier
dt  = DecisionTreeClassifier(max_depth=10, random_state=42)
rf  = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=42)

model = VotingClassifier(
    estimators=[('dt', dt), ('rf', rf), ('xgb', xgb)],
    voting='soft'
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc*100:.2f}%")

# SHAP on XGB
explainer = shap.TreeExplainer(model.estimators_[2])
shap_values = explainer.shap_values(X_train)
print("SHAP done")

# Load extra datasets
desc       = pd.read_csv('symptom_Description.csv')
precaution = pd.read_csv('symptom_precaution.csv')
severity   = pd.read_csv('Symptom-severity.csv')

desc.columns       = desc.columns.str.strip()
precaution.columns = precaution.columns.str.strip()
severity.columns   = severity.columns.str.strip()

desc['Disease']       = desc['Disease'].str.strip()
precaution['Disease'] = precaution['Disease'].str.strip()
severity['Symptom']   = severity['Symptom'].str.strip()

joblib.dump({
    'model': model, 'le': le, 'symptoms': SYMPTOMS,
    'explainer': explainer, 'shap_values': shap_values,
    'description': desc, 'precaution': precaution, 'severity': severity
}, 'model.pkl')
print("Saved model.pkl")
