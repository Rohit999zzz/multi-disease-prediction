# Disease Predictor (Explainable AI)

This is a Flask web application that predicts diseases based on natural language symptom descriptions. The app uses a machine learning ensemble model and provides explainable AI (SHAP) visualizations for its predictions.

## Features

- Enter symptoms in natural language (e.g., "I have headache and chest pain")
- Extracts recognized symptoms using Gemini AI
- Predicts the most likely disease using a trained ensemble model
- Shows SHAP-based explanation for the prediction

## Project Structure

- `app.py` — Main Flask application
- `ensemble_model.pkl` — Trained ensemble model
- `label_encoder.pkl` — Label encoder for disease names
- `symptom_list.pkl` — List of recognized symptoms
- `requirements.txt` — Python dependencies
- `static/shap_explanation.png` — SHAP explanation image (generated at runtime)
- `templates/index.html` — HTML template for the web interface

## Setup Instructions

1. **Clone the repository and navigate to the project folder.**

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Add your Gemini API key:**
   - Replace the API key in `app.py`:
     ```python
     genai.configure(api_key="YOUR_API_KEY_HERE")
     ```

4. **Ensure the following files are present:**
   - `ensemble_model.pkl`
   - `label_encoder.pkl`
   - `symptom_list.pkl`

5. **Run the application:**
   ```sh
   python app.py
   ```

6. **Open your browser and go to:**
   ```
   http://127.0.0.1:5000/
   ```

## Notes

- The SHAP explanation image is generated for each prediction and saved as `static/shap_explanation.png`.
- The application requires internet access for Gemini API calls.

## License

This project is for educational purposes.
