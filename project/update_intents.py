import json

with open("intents.json", "r") as f:
    data = json.load(f)

for intent in data["intents"]:
    if intent["tag"] == "question_papers_2nd_year":
        # Add new patterns
        patterns = set(intent["patterns"])
        patterns.update([
            "Question Papers - 2nd Year",
            "what question papers are available for 2nd year",
            "2nd year materials",
            "2nd year question papers and notes",
            "notes for 2nd year"
        ])
        intent["patterns"] = list(patterns)
        intent["responses"] = [
            "These are the question papers and notes available in our collections for the 2nd Year:<br>- <b>Analog Electronics Circuits (AEC)</b>.<br>You can ask for 'AEC question paper' or 'AEC notes' to download them!"
        ]
    elif intent["tag"] == "question_papers_aec_2nd_year":
        # Add a lot more robust patterns
        patterns = set(intent["patterns"])
        patterns.update([
            "give me aec notes",
            "can you please give me the aec question paper",
            "i need aec notes",
            "aec notes please",
            "can you please give me the aec notes",
            "analog electronics circuits question paper",
            "provide aec notes",
            "where is aec question paper",
            "aec previous year papers",
            "i want aec question paper",
            "aec internals",
            "aec question papers and notes",
            "can i have aec question paper",
            "aec pdf",
            "aec notes pdf"
        ])
        intent["patterns"] = list(patterns)

with open("intents.json", "w") as f:
    json.dump(data, f, indent=2)

print("intents.json updated successfully.")
