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
