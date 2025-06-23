from flask import Flask, request, render_template
import joblib
import google.generativeai as genai
import os
import pandas as pd
import re
import shap
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Load model and assets
model = joblib.load("ensemble_model.pkl")
symptoms = joblib.load("symptom_list.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Setup Gemini API
genai.configure(api_key="API_KEY")
gemini = genai.GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html", prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form.get("user_symptoms")

    # Gemini Prompt
    prompt = f"""You are a medical assistant. From the following user input, identify the symptoms from this list: {symptoms}.
User input: '{user_input}'.
Return ONLY a Python list of matched symptom strings, nothing else.
Format: ['symptom1', 'symptom2']"""

    response = gemini.generate_content(prompt)
    try:
        response_text = response.text.strip()
        response_text = re.sub(r'```python|```', '', response_text)
        match = re.search(r'\[(.*?)\]', response_text)
        extracted_symptoms = [s.strip().strip("'\"") for s in match.group(1).split(',')] if match else []
        extracted_symptoms = [s for s in extracted_symptoms if s in symptoms]
    except Exception as e:
        print(f"Extraction error: {e}")
        extracted_symptoms = []

    input_vector = [1 if sym in extracted_symptoms else 0 for sym in symptoms]
    input_df = pd.DataFrame([input_vector], columns=symptoms)

    # Predict
    pred = model.predict(input_df)[0]
    disease = label_encoder.inverse_transform([pred])[0]

    # SHAP Explanation (using TreeExplainer on a base RF estimator)
    try:
        base_estimator = model.estimators_[0]  # Use one RF or similar model
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer.shap_values(input_df)

        plt.clf()
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[pred][0],
                base_values=explainer.expected_value[pred],
                data=input_df.iloc[0].values,
                feature_names=symptoms
            ),
            show=False
        )
        shap_image_path = os.path.join("static", "shap_explanation.png")
        plt.savefig(shap_image_path, bbox_inches='tight', dpi=100)
        plt.close()
    except Exception as e:
        print(f"SHAP error: {e}")
        shap_image_path = None

    return render_template("index.html",
                           prediction=disease,
                           input_text=user_input,
                           extracted=extracted_symptoms,
                           total_symptoms=len(extracted_symptoms),
                           shap_image=shap_image_path)

if __name__ == '__main__':
    app.run(debug=True)
