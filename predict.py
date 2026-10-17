import joblib

model = joblib.load("notice_classifier.joblib")

while True:
    notice = input("\nEnter a notice (or type 'exit'): ")

    if notice.lower() == "exit":
        break

    category = model.predict([notice])[0]
    probabilities = model.predict_proba([notice])[0]

    confidence = max(probabilities)

    print("Predicted category:", category)
    print("Model confidence:", round(confidence * 100, 2), "%")