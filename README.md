# Comment Toxicity Prediction

## 📌 Overview
This project focuses on predicting the toxicity level of user comments using Natural Language Processing (NLP) and Deep Learning. The model is trained on a dataset containing various types of toxic comments and classifies them into multiple categories.

## 🔧 Technologies Used
- Python 🐍
- TensorFlow/Keras 🔥
- Pandas 🏛️
- Scikit-Learn 📊
- LSTM & Bidirectional LSTM

## 🚀 Features
- Text preprocessing using **TextVectorization**
- Deep Learning model with **LSTM** and **Bidirectional LSTM** layers
- Multi-label classification for six toxicity categories
- Model trained and evaluated using TensorFlow

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

## 🔍 Prediction Example
```python
# Load model
model = tf.keras.models.load_model('Toxicity.h5')

# Sample prediction
input_text = vectorize(['U are ugly'])
res = model.predict(tf.expand_dims(input_text, 0))

# Display results
for idx, column in enumerate(df.columns[2:]):
    print(f"{column}: {res[0][idx] > 0.5}")
```

## 📊 Results
After training, the model can classify comments into multiple toxicity categories with reasonable accuracy.

### Example Output:
```
Toxic: True
Severe Toxic: False
Obscene: False
Threat: False
Insult: True
Identity Hate: False
```

## 🎯 Future Improvements
- Improve accuracy by experimenting with **transformers (BERT, DistilBERT)**
- Fine-tune hyperparameters for better performance
- Deploy the model using **Flask, FastAPI, or Streamlit**

## 📌 Contributing
Feel free to fork this repository, suggest improvements, and contribute! 🚀

## 📝 License
This project is open-source and available under the MIT License.

