import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import send_from_directory


model = tf.keras.models.load_model('Toxicity.h5')

df = pd.read_csv('train.csv')  
toxicity_labels = df.columns[2:] 

MAX_FEATURES = 200000
vectorize = TextVectorization(max_tokens=MAX_FEATURES, output_sequence_length=2000, output_mode='int')
vectorize.adapt(df['comment_text'].values)

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('text', '')

    if not input_text:
        return jsonify({'error': 'No text provided'}), 400

    input_vector = vectorize([input_text])
    res = model.predict(input_vector)[0]

    results = {label: bool(pred > 0.5) for label, pred in zip(toxicity_labels, res)}

    return jsonify({'input_text': input_text, 'toxicity_labels': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
