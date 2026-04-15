import os
import random
import json
import traceback

# Restrict CPU threads before importing torch — reduces RAM usage on free tier
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from flask import Flask, render_template, request, jsonify
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "intents.json")
MODEL_PATH   = os.path.join(BASE_DIR, "data.pth")

# ── Load intents ONCE at startup (avoid re-reading file on every message) ──────
with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents_data = json.load(f)

# ── Load model ONCE at startup, force CPU so no GPU memory is allocated ────────
_data       = torch.load(MODEL_PATH, weights_only=False, map_location="cpu")
input_size  = _data["input_size"]
hidden_size = _data["hidden_size"]
output_size = _data["output_size"]
all_words   = _data["all_words"]
tags        = _data["tags"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(_data["model_state"])
model.eval()

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)


# ── Fallback: TF-IDF cosine similarity when model confidence is low ────────────
def fallback_response(user_input):
    all_patterns = []
    tags_list    = []

    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            all_patterns.append(pattern)
            tags_list.append(intent["tag"])

    vectorizer = TfidfVectorizer()
    X          = vectorizer.fit_transform(all_patterns + [user_input])
    similarity = cosine_similarity(X[-1], X[:-1])
    best_tag   = tags_list[similarity.argmax()]

    for intent in intents_data["intents"]:
        if intent["tag"] == best_tag:
            return random.choice(intent["responses"])

    return "I didn't understand that. Please try rephrasing."


# ── Core chatbot response function ────────────────────────────────────────────
def get_response(user_input):
    sentence = tokenize(user_input)
    X        = bag_of_words(sentence, all_words)
    X        = torch.from_numpy(X.reshape(1, -1))

    output       = model(X)
    _, predicted = torch.max(output, dim=1)
    tag          = tags[predicted.item()]
    prob         = torch.softmax(output, dim=1)[0][predicted.item()].item()

    if prob > 0.5:
        for intent in intents_data["intents"]:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])
                # Handle structured faculty-list responses
                if isinstance(response, dict) and "faculty" in response:
                    return "".join(
                        f"Name: {f['name']}<br>"
                        f"Qualification: {f['qualification']}<br>"
                        f"Designation: {f['designation']}<br><br>"
                        for f in response["faculty"]
                    )
                return response

    return fallback_response(user_input)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def chat():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def handle_message():
    try:
        message       = request.form["msg"]
        response_text = get_response(message)
        return jsonify({"response_text": response_text})
    except Exception as e:
        print("Backend Error:", traceback.format_exc())
        return jsonify({"response_text": f"Sorry, something went wrong: {str(e)}"})


# ── Entry point (python app.py) ────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)