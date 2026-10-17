# NoticeMate ML – Academic Notice Classifier

NoticeMate ML is a machine learning project that automatically classifies academic notices into predefined categories using Natural Language Processing (NLP) and text classification techniques.

The model helps students identify the type of notice quickly, such as placement announcements, examination updates, scholarships, and college events.

## Features

- Automatically predicts the category of an academic notice.
- Accepts a notice title and content as input.
- Uses TF-IDF to convert text into numerical features.
- Uses a Linear Support Vector Machine (Linear SVM) for classification.
- Provides predictions through a Python Flask REST API.
- Supports integration with a web application.

## Notice Categories

The model classifies notices into eight categories:

1. Academics
2. Event
3. Examination
4. General
5. Holiday
6. Placement
7. Scholarship
8. Sports

## Tech Stack

- **Language:** Python
- **Machine Learning:** Scikit-learn
- **NLP:** TF-IDF Vectorization
- **Classification Algorithm:** Linear SVM
- **Backend API:** Flask
- **Data Processing:** Pandas
- **Model Serialization:** Joblib

## Project Structure

```text
NotiX_ml/
├── app.py
├── notice_classifier.joblib
├── requirements.txt
├── train_model.py
├── test_model.py
├── notices.csv
└── notices_shuffled.csv
```

*The structure above represents the main project files. Include only files that actually exist in your repository.*

## How It Works

1. The user submits a notice title and content.
2. The Flask API receives the input.
3. The saved ML pipeline preprocesses the text using TF-IDF.
4. The Linear SVM predicts the most likely category.
5. The API returns the predicted category as JSON.

## Model Training

The model was trained using a labeled dataset of academic notices.

The training pipeline uses:

- TF-IDF vectorization with unigrams and bigrams.
- Lowercasing and sublinear term-frequency scaling.
- Linear SVM classification.
- Stratified train-test splitting.
- Accuracy, precision, recall, F1-score, and confusion matrix evaluation.

To train the model again, activate your virtual environment, install the dependencies, and run:

```bash
python train_model.py
```

Ensure that the training script points to the correct dataset and saves the output as `notice_classifier.joblib`.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tanvi-Kuwar/NotiX_ml.git
cd NotiX_ml
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask API

```bash
python app.py
```

The local API will run at:

`http://127.0.0.1:5000`

## API Usage

### Endpoint

`POST /predict`

### Request

Send a JSON request containing the notice title and content.

```json
{
  "title": "Campus Placement Drive",
  "content": "Eligible students must register for the recruitment process."
}
```

### Example Response

```json
{
  "category": "Placement"
}
```

The response shown is an example; the actual category depends on the model prediction.

## Model Evaluation

During development, the Linear SVM achieved approximately **96.4% accuracy** and a **0.964 macro F1-score** on the held-out test split.

A separate manually prepared test set achieved 15 correct predictions out of 16 (93.75%).

These are development results, not a guarantee of real-world performance. The manual test set is small, and performance may vary on new or ambiguous notices.

## Deployment

The Flask API can be deployed to a compatible hosting platform, including Vercel, subject to its current Python runtime, dependency, function-size, and execution limits.

After deployment, update the frontend or parent backend to call the deployed API URL instead of the local address.

## Future Improvements

- Improve classification of similar categories, especially Event and Sports.
- Expand and diversify the labeled dataset.
- Evaluate performance on additional unseen notices.
- Integrate the classifier with the main NoticeMate application.
- Add notice storage, filtering, and category-based search.

## Author

**Tanvi Kuwar**

GitHub: [Tanvi-Kuwar](https://github.com/Tanvi-Kuwar)

---

*NoticeMate ML is an academic machine learning project focused on automating the categorization of college notices.*
