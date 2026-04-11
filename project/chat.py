# Import the required libraries
from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import speech_recognition as sr
import pyttsx3

# Initialize Flask app
app = Flask(__name__)

# Load intents for the chatbot
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the pre-trained model and its data
FILE = "data.pth"
data = torch.load(FILE)
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

# Initialize speech recognition
recognizer = sr.Recognizer()

# Function to get response from chatbot
def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

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

# Route for handling speech input
@app.route('/speech', methods=['POST'])
def handle_speech():
    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = recognizer.listen(source)

    try:
        speech_text = recognizer.recognize_google(audio)
        response = get_response(speech_text)
        return jsonify({'response_text': response, 'response_speech': speech_text})
    except:
        return jsonify({'response_text': "Sorry, I couldn't understand that.", 'response_speech': ""})

if __name__ == '__main__':
    app.run(debug=True)
