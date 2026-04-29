import os
import random
import json
import traceback
from datetime import datetime

# Restrict CPU threads before importing torch — reduces RAM usage on free tier
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

from flask import Flask, render_template, request, jsonify, abort
from werkzeug.utils import secure_filename
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import wikipedia

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH      = os.path.join(BASE_DIR, "intents.json")
MODEL_PATH        = os.path.join(BASE_DIR, "data.pth")
NOTES_DATA_PATH   = os.path.join(BASE_DIR, "notes_data.json")
UPLOAD_BASE       = os.path.join(BASE_DIR, "static", "notes", "uploaded")
ALLOWED_EXT       = {"pdf", "doc", "docx", "ppt", "pptx", "txt"}

# ── Teacher password (change this to a strong secret) ───────────────────────
TEACHER_PASSWORD  = "ssit@teacher2026"

# ── Year tag mapping: chatbot tag → notes_data.json key ──────────────────────
YEAR_TAG_MAP = {
    "question_papers_1st_year": "1st year",
    "question_papers_2nd_year": "2nd year",
    "question_papers_3rd_year": "3rd year",
    "question_papers_4th_year": "4th year",
}

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


# ── Load / save dynamic notes JSON ────────────────────────────────────────────
def load_notes_data():
    """Read notes_data.json — returns dict with year keys."""
    try:
        with open(NOTES_DATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"1st year": [], "2nd year": [], "3rd year": [], "4th year": []}


def save_notes_data(data):
    """Persist notes_data.json safely."""
    with open(NOTES_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def build_dynamic_notes_html(year_key):
    """
    Build HTML for dynamically added notes.
    Only the LAST entry is shown as 'Newly Added'; older ones appear as regular notes.
    """
    notes = load_notes_data().get(year_key, [])
    if not notes:
        return ""

    def note_li(note):
        subject  = note.get("subject", "General")
        content  = note.get("content", "")
        added_on = note.get("added_on", "")
        li = ""
        if content.startswith("http") or content.startswith("/"):
            fname = content.rsplit("/", 1)[-1]
            li = f'<li><strong>{subject}:</strong>&nbsp;<a href="{content}" target="_blank" style="color:#c4b5fd;">{fname} — Click to open</a>'
        else:
            li = f'<li><strong>{subject}:</strong>&nbsp;{content}'
        if added_on:
            li += f'&nbsp;<span style="font-size:0.75rem;color:#64748b;">({added_on})</span>'
        li += '</li>'
        return li

    html = '<div class="dynamic-notes-section">'

    # Older notes (all except the last)
    if len(notes) > 1:
        html += '<h4 style="color:#a78bfa;margin-bottom:6px;border-top:1px solid rgba(255,255,255,0.1);padding-top:10px;">📚 Added Notes</h4>'
        html += '<ul class="notes-list">'
        for note in notes[:-1]:
            html += note_li(note)
        html += '</ul>'

    # Latest note — highlighted as Newly Added
    html += '<h4 style="color:#10b981;margin-bottom:6px;margin-top:8px;">🆕 Newly Added</h4>'
    html += '<ul class="notes-list">'
    html += note_li(notes[-1])
    html += '</ul></div>'
    return html


# ── Fallback: TF-IDF cosine similarity when model confidence is low ────────────
def fallback_response(user_input):
    all_patterns = []
    tags_list    = []

    for intent in intents_data["intents"]:
        for pattern in intent["patterns"]:
            all_patterns.append(pattern)
            tags_list.append(intent["tag"])

    vectorizer = TfidfVectorizer(stop_words='english')
    X          = vectorizer.fit_transform(all_patterns + [user_input])
    similarity = cosine_similarity(X[-1], X[:-1])
    
    max_sim = similarity.max()
    if max_sim < 0.2:
        return "I am sorry, I am just an academic helper bot and I do not have an answer for that. Can you please rephrase or ask something else?"

    best_tag   = tags_list[similarity.argmax()]

    for intent in intents_data["intents"]:
        if intent["tag"] == best_tag:
            return random.choice(intent["responses"])

    return "I didn't fully understand your question. Can you please explain it again?"


# ── Core chatbot response function ────────────────────────────────────────────
def get_response(user_input):
    lower_input = user_input.lower().strip()
    
    # Removed hardcoded conversational rules to allow NN prediction
        
    # 6. Learning Behavior Rule: Extend knowledge
    if lower_input.startswith("learn:"):
        parts = user_input[6:].split("|")
        if len(parts) == 2:
            question = parts[0].strip()
            answer = parts[1].strip()
            if question and answer:
                new_intent = {
                    "tag": f"learned_{random.randint(10000, 99999)}",
                    "patterns": [question],
                    "responses": [answer]
                }
                intents_data["intents"].append(new_intent)
                try:
                    with open(INTENTS_PATH, "w", encoding="utf-8") as f:
                        json.dump(intents_data, f, indent=4)
                    return f"Thank you! I have learned the answer to: '{question}'"
                except Exception as e:
                    return "I had trouble saving the new knowledge. Data safety rule enforced."
        return "To teach me, use this format: 'Learn: <question> | <answer>'"

    sentence = tokenize(user_input)
    X        = bag_of_words(sentence, all_words)
    X        = torch.from_numpy(X.reshape(1, -1))

    output       = model(X)
    _, predicted = torch.max(output, dim=1)
    tag          = tags[predicted.item()]
    prob         = torch.softmax(output, dim=1)[0][predicted.item()].item()

    if prob > 0.75:
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
                # Append dynamic notes if this is a year-based notes response
                if tag in YEAR_TAG_MAP:
                    year_key    = YEAR_TAG_MAP[tag]
                    dynamic_html = build_dynamic_notes_html(year_key)
                    if dynamic_html:
                        response = str(response) + dynamic_html
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


# ── Teacher: Add Notes (supports file upload + plain text) ───────────────────
@app.route("/add_note", methods=["POST"])
def add_note():
    """Password-protected endpoint for teachers to add notes or upload files."""
    try:
        is_multipart = request.content_type and "multipart" in request.content_type

        if is_multipart:
            password = request.form.get("password", "")
            year     = request.form.get("year", "").strip()
            subject  = request.form.get("subject", "").strip()
            content  = request.form.get("content", "").strip()
            file     = request.files.get("file")
        else:
            payload  = request.get_json(force=True)
            password = payload.get("password", "")
            year     = payload.get("year", "").strip()
            subject  = payload.get("subject", "").strip()
            content  = payload.get("content", "").strip()
            file     = None

        # ── Auth check ─────────────────────────────────────────────────────────
        if password != TEACHER_PASSWORD:
            return jsonify({"success": False, "error": "Incorrect password. Access denied."}), 403

        # ── Validate year / subject ────────────────────────────────────────────
        valid_years = ["1st year", "2nd year", "3rd year", "4th year"]
        if year not in valid_years:
            return jsonify({"success": False, "error": "Invalid year selected."}), 400
        if not subject:
            return jsonify({"success": False, "error": "Subject cannot be empty."}), 400

        # ── Handle file upload ─────────────────────────────────────────────────
        file_url = None
        if file and file.filename:
            ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
            if ext not in ALLOWED_EXT:
                return jsonify({"success": False, "error": "Only PDF, DOC, DOCX, PPT, PPTX, TXT files allowed."}), 400
            safe_name   = secure_filename(file.filename)
            year_folder = os.path.join(UPLOAD_BASE, year.replace(" ", "_"))
            os.makedirs(year_folder, exist_ok=True)
            file.save(os.path.join(year_folder, safe_name))
            file_url = f"/static/notes/uploaded/{year.replace(' ', '_')}/{safe_name}"
            content  = file_url   # store the URL as content

        if not content:
            return jsonify({"success": False, "error": "Provide notes text/link OR upload a file."}), 400

        # ── Append to notes_data.json ──────────────────────────────────────────
        notes_data = load_notes_data()
        notes_data[year].append({
            "subject":  subject,
            "content":  content,
            "type":     "file" if file_url else "text",
            "added_on": datetime.now().strftime("%d %b %Y")
        })
        save_notes_data(notes_data)

        return jsonify({"success": True, "message": f"'{subject}' added to {year} successfully!"})

    except Exception as e:
        print("Add Note Error:", traceback.format_exc())
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


# ── Teacher: Get notes list for a year (used by delete UI) ────────────────────
@app.route("/get_notes", methods=["POST"])
def get_notes():
    try:
        payload  = request.get_json(force=True)
        password = payload.get("password", "")
        year     = payload.get("year", "").strip()
        if password != TEACHER_PASSWORD:
            return jsonify({"success": False, "error": "Incorrect password."}), 403
        valid_years = ["1st year", "2nd year", "3rd year", "4th year"]
        if year not in valid_years:
            return jsonify({"success": False, "error": "Invalid year."}), 400
        notes = load_notes_data().get(year, [])
        return jsonify({"success": True, "notes": notes})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ── Teacher: Delete a note (only from notes_data.json, never touches intents) ─────
@app.route("/delete_note", methods=["POST"])
def delete_note():
    try:
        payload  = request.get_json(force=True)
        password = payload.get("password", "")
        year     = payload.get("year", "").strip()
        index    = payload.get("index", -1)
        if password != TEACHER_PASSWORD:
            return jsonify({"success": False, "error": "Incorrect password."}), 403
        valid_years = ["1st year", "2nd year", "3rd year", "4th year"]
        if year not in valid_years:
            return jsonify({"success": False, "error": "Invalid year."}), 400
        notes_data = load_notes_data()
        year_notes = notes_data.get(year, [])
        if not isinstance(index, int) or index < 0 or index >= len(year_notes):
            return jsonify({"success": False, "error": "Invalid note index."}), 400
        deleted = year_notes.pop(index)
        # Also remove uploaded file from disk if it was a file upload
        content = deleted.get("content", "")
        if content.startswith("/static/notes/uploaded/"):
            file_path = os.path.join(BASE_DIR, content.lstrip("/").replace("/", os.sep))
            if os.path.exists(file_path):
                os.remove(file_path)
        notes_data[year] = year_notes
        save_notes_data(notes_data)
        return jsonify({"success": True, "message": "Note deleted successfully."})
    except Exception as e:
        print("Delete Note Error:", traceback.format_exc())
        return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500


# ── Entry point (python app.py) ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)