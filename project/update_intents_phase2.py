import json

with open("intents.json", "r") as f:
    data = json.load(f)

# Update the 2nd year general query response
for intent in data["intents"]:
    if intent["tag"] == "question_papers_2nd_year":
        intent["responses"] = [
            "We have fresh updates for the 2nd Year!<br><br>"
            "📚 <b>Available Question Papers & Notes:</b><br>"
            "🔹 Analog Electronics Circuits (AEC)<br>"
            "🔹 Digital Electronics Circuits (DEC)<br>"
            "🔹 Signals and Systems (SAS)<br><br>"
            "<b>To download them, simply type exactly:</b><br>"
            "👉 <i>'give me aec notes'</i><br>"
            "👉 <i>'give me dec notes'</i><br>"
            "👉 <i>'give me sas notes'</i>"
        ]

# Create intent for DEC
dec_intent = {
    "tag": "question_papers_dec",
    "patterns": [
        "give me dec notes",
        "i need dec notes",
        "dec notes please",
        "dec question paper",
        "give me dec question paper",
        "digital electronics circuits notes",
        "digital electronics question paper",
        "dec internals",
        "dec previous year papers"
    ],
    "responses": [
        "Here are the latest DEC 2023 Internals and SEE question papers: <a href='/static/DEC_2023_internals_and_SEE_question_papers.pdf' target='_blank' style='color:#8b5cf6; text-decoration:underline; font-weight:bold;'>Download DEC PDF</a>."
    ],
    "context_set": ""
}

# Create intent for SAS
sas_intent = {
    "tag": "question_papers_sas",
    "patterns": [
        "give me sas notes",
        "i need sas notes",
        "sas notes please",
        "sas question paper",
        "give me sas question paper",
        "signals and systems notes",
        "signals and systems question paper",
        "sas internals",
        "sas previous year papers"
    ],
    "responses": [
        "Here are the latest SAS 2023 Internals and SEE question papers: <a href='/static/SAS_2023_internals_and_SEE_question_papers.pdf' target='_blank' style='color:#8b5cf6; text-decoration:underline; font-weight:bold;'>Download SAS PDF</a>."
    ],
    "context_set": ""
}

# Create intent for 3rd Year button
yr3_intent = {
    "tag": "question_papers_3rd_year",
    "patterns": [
        "Question Papers - 3rd Year",
        "what question papers are available for 3rd year",
        "3rd year materials",
        "3rd year question papers and notes",
        "notes for 3rd year"
    ],
    "responses": [
        "Currently, there are no fresh question papers added today for the 3rd Year. We are organizing the CCN and other notes, please check back soon!"
    ],
    "context_set": ""
}

# Safely append intents if they don't already exist
tags_to_add = [dec_intent, sas_intent, yr3_intent]
existing_tags = [intent["tag"] for intent in data["intents"]]

for new_intent in tags_to_add:
    if new_intent["tag"] in existing_tags:
        # Update existing
        for intent in data["intents"]:
            if intent["tag"] == new_intent["tag"]:
                intent["patterns"] = new_intent["patterns"]
                intent["responses"] = new_intent["responses"]
    else:
        # Add new
        data["intents"].append(new_intent)

with open("intents.json", "w") as f:
    json.dump(data, f, indent=2)

print("intents.json updated successfully.")
