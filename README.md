# Comment Toxicity Prediction

## 📌 Overview
This project focuses on predicting the toxicity level of user comments using Natural Language Processing (NLP) and Deep Learning. The model is trained on a dataset containing various types of toxic comments and classifies them into multiple categories.

## 🔧 Technologies Used
- Python 🐍
- TensorFlow/Keras 🔥
- Pandas 🏛️
- Scikit-Learn 📊
- Flask 🌍
- LSTM & Bidirectional LSTM

## 🚀 Features
- Text preprocessing using **TextVectorization**
- Deep Learning model with **LSTM** and **Bidirectional LSTM** layers
- Multi-label classification for six toxicity categories
- Model trained and evaluated using TensorFlow
- **Flask API** for real-time predictions

## 📂 Dataset
The model is trained on the **Jigsaw Toxic Comment Classification Challenge** dataset, which includes the following toxicity labels:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## 🛠 Installation
To set up the environment and run the project, install the required dependencies:
```bash
pip install matplotlib tensorflow pandas scikit-learn
```

## 📜 Model Training
```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, LSTM, Bidirectional, Dense, Embedding
from tensorflow.keras.models import Sequential

# Load dataset
df = pd.read_csv('train.csv')
x = df['comment_text']
y = df[df.columns[2:]].values

# Text Vectorization
MAX_FEATURES = 200000
vectorize = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=2000, output_mode='int')
vectorize.adapt(x.values)
vectorize_text = vectorize(x.values)

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((vectorize_text, y)).shuffle(160000).batch(16).prefetch(8)
train = dataset.take(int(len(dataset) * 0.7))
val = dataset.skip(int(len(dataset) * 0.7)).take(int(len(dataset) * 0.2))
test = dataset.skip(int(len(dataset) * 0.9)).take(int(len(dataset) * 0.1))

# Model Definition
model = Sequential([
    Embedding(MAX_FEATURES + 1, 32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])

# Compile Model
model.compile(loss='BinaryCrossentropy', optimizer='Adam')

# Train Model
history = model.fit(train, epochs=8, validation_data=val)

# Save Model
model.save('Toxicity.h5')
```

## Flask API Deployment 🌍
The model is now deployed using Flask, allowing real-time text predictions via an API.
```python
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from flask import Flask, request, jsonify

# Load trained model
model = tf.keras.models.load_model('Toxicity.h5')

# Load dataset to get label names
df = pd.read_csv('train.csv')  
toxicity_labels = df.columns[2:]  # Assuming first two columns are not labels

# Define Text Vectorization (same as used in training)
MAX_FEATURES = 200000
vectorize = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=2000, output_mode='int')
vectorize.adapt(df['comment_text'].values)  # Adapt vectorizer to training data

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    # Vectorize input
    input_vector = vectorize([input_text])
    res = model.predict(input_vector)[0]

    # Format output (convert predictions to True/False)
    results = {label: bool(pred > 0.5) for label, pred in zip(toxicity_labels, res)}

    return jsonify({'input_text': input_text, 'toxicity_labels': results})

if __name__ == '__main__':
    app.run(debug=True)
```

## 🚀 Running the Flask API
Run the following command to start the server:
```bash
python app.py
```
You can send a request to the API using Postman 
### Example Output:
```json
{
  "input_text": "You are so dumb!",
  "toxicity_labels": {
    "toxic": true,
    "severe_toxic": false,
    "obscene": false,
    "threat": false,
    "insult": true,
    "identity_hate": false
  }
}
```

## 🎯 Future Improvements
- Improve accuracy by experimenting with **transformers (BERT, DistilBERT)**
- Fine-tune hyperparameters for better performance
- Deploy the model using **FastAPI, Docker or Cloud platforms**

## 📌 Contributing
Feel free to fork this repository, suggest improvements, and contribute! 🚀

## 📝 License
This project is open-source and available under the MIT License.

