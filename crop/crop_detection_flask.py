from flask import Flask, request, jsonify, render_template
import requests
import google.generativeai as genai

app = Flask(__name__)

# Custom Vision API settings
CUSTOM_VISION_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/e507e178-1390-468a-9c8c-81f228dcadcc/classify/iterations/crop_diseases_detection/image"
PREDICTION_KEY = "c43480edd8d047ab8546b37fea8e89c9"

HEADERS = {
    "Content-Type": "application/octet-stream",
    "Prediction-Key": PREDICTION_KEY
}

# Gemini API settings
GEMINI_API_KEY = "AIzaSyDWN-lXdhrNSD4arKrFA6d581eKKz0iK8c"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def get_cure_recommendation(disease_name):
    """Get treatment recommendations using Gemini LLM"""
    prompt = f"The plant has been diagnosed with {disease_name}. Provide a recommended cure, including organic and chemical treatment options. As plain text not markdown"
    response = model.generate_content(prompt)
    return response.text if response else "No recommendation available."

@app.route("/")
def index():
    return render_template("crop_disease_detection.html") 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    image_data = file.read()

    # Call Azure Custom Vision API
    response = requests.post(CUSTOM_VISION_ENDPOINT, headers=HEADERS, data=image_data)
    
    if response.status_code != 200:
        return jsonify({"error": "Prediction failed", "details": response.text}), response.status_code

    prediction_result = response.json()
    
    # Extract the top predicted disease name
    predictions = prediction_result.get("predictions", [])
    if not predictions:
        return jsonify({"error": "No prediction results"}), 400

    top_prediction = max(predictions, key=lambda x: x['probability'])
    disease_name = top_prediction['tagName']
    
    # Get cure recommendation from Gemini
    cure_recommendation = get_cure_recommendation(disease_name)

    return jsonify({
        "disease": disease_name,
        "probability": top_prediction["probability"],
        "cure_recommendation": cure_recommendation
    })

if __name__ == '__main__':
    app.run(debug=True)
