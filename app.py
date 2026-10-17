
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model only once
model = joblib.load("notice_classifier.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "NoticeMate ML API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Request must contain JSON"}), 400

    title = str(data.get("title", "")).strip()
    content = str(data.get("content", "")).strip()

    if not title and not content:
        return jsonify({
            "error": "Please provide a notice title or content"
        }), 400

    # Keep the input format consistent with training:
    # title + content
    text = title + " " + content

    prediction = model.predict([text])[0]

    return jsonify({
        "title": title,
        "category": str(prediction)
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
