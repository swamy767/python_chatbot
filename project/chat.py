import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Import the required libraries
from flask import Flask, render_template, request, jsonify
import random
import json
import torch
torch.set_num_threads(1)
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
def fallback_response(user_input):
    all_patterns = []
    tags_list = []

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            all_patterns.append(pattern)
            tags_list.append(intent['tag'])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_patterns + [user_input])

    similarity = cosine_similarity(X[-1], X[:-1])
    index = similarity.argmax()

    best_tag = tags_list[index]

    for intent in intents['intents']:
        if intent['tag'] == best_tag:
            return random.choice(intent['responses'])

    return "I didn't understand."
# Initialize Flask app
app = Flask(__name__)

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load intents for the chatbot
with open(os.path.join(BASE_DIR, 'intents.json'), 'r') as json_data:
    intents = json.load(json_data)

# Load the pre-trained model and its data
FILE = os.path.join(BASE_DIR, "data.pth")
data = torch.load(FILE, weights_only=False)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

# Function to get response from chatbot
def get_response(user_input):
    sentence = tokenize(user_input)
    
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                if isinstance(response, dict) and "faculty" in response:
                    response_text = ""
                    for f in response["faculty"]:
                        response_text += f"Name: {f['name']}<br>"
                        response_text += f"Qualification: {f['qualification']}<br>"
                        response_text += f"Designation: {f['designation']}<br><br>"
                    return response_text
                return response
    else:
        return fallback_response(user_input)

# Route for serving the chat interface
@app.route('/')
def chat():
    return render_template('index.html')

# Route for handling chat messages
@app.route('/get', methods=['POST'])
def handle_message():
    message = request.form['msg']
    response_text = get_response(message)
    return jsonify({'response_text': response_text})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    from waitress import serve
    print(f"Starting production server on port {port}...")
    serve(app, host="0.0.0.0", port=port)