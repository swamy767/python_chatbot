import os
import shutil
import json
import re
from urllib.parse import quote
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, 'intents.json')

SOURCE_DIR = r"C:\Users\LENOVO\OneDrive\Desktop\Question papre and notes"
STATIC_DIR = r"static\notes"

os.makedirs(STATIC_DIR, exist_ok=True)

SUBJECTS = [
    "AC", "VLSI", "DSP", "CS", "ME", "MC", "Network", 
    "EVS", "Management", "Control System", "Micro processor", "Opearational Amplifier"
]

def determine_subject(filename, folder):
    name_lower = filename.lower()
    
    if "question paper" in name_lower or "internals" in name_lower or "see " in name_lower:
        return "Question Papers"
    
    for sub in SUBJECTS:
        if sub.lower() in name_lower:
            return sub
            
    if "presentation" in folder.lower():
        if "report" in name_lower: return "Reports"
        if "abstract" in name_lower: return "Abstracts"
        if "ppt" in name_lower or "seminar" in name_lower: return "Seminars/PPTs"
        return "Documentation"
        
    return "Others"

def process_folder(folder_name, list_id_prefix):
    source_path = os.path.join(SOURCE_DIR, folder_name)
    target_path = os.path.join(STATIC_DIR, folder_name)
    
    if not os.path.exists(source_path):
        return ""
        
    os.makedirs(target_path, exist_ok=True)
    
    grouped_files = {}
    
    for filename in os.listdir(source_path):
        filepath = os.path.join(source_path, filename)
        if os.path.isfile(filepath):
            dest_file = os.path.join(target_path, filename)
            if not os.path.exists(dest_file):
                shutil.copy2(filepath, dest_file)
                
            subject = determine_subject(filename, folder_name)

            # User explicitly requested to omit VLSI and DSP from 2nd year
            if folder_name == "2nd year" and subject in ["VLSI", "DSP"]:
                continue

            if subject not in grouped_files:
                grouped_files[subject] = []
            
            web_path = f"/static/notes/{quote(folder_name)}/{quote(filename)}"
            clean_name = os.path.splitext(filename)[0]
            grouped_files[subject].append({"name": clean_name, "url": web_path})
    
    subject_keys = sorted(grouped_files.keys(), key=lambda k: "0" + k if k == "Question Papers" else k)
    
    html = f"""
    <div id="{list_id_prefix}" class="notes-list-container">
        <input type="text" class="notes-search-input" onkeyup="filterNotes(this, '{list_id_prefix}')" placeholder="Search files... (e.g. AC)">
    """
    
    for subj in subject_keys:
        html += f"""
        <div class="subject-section" data-subject="{subj}">
            <h4>{subj}</h4>
            <ul class="notes-list">
        """
        for item in sorted(grouped_files[subj], key=lambda x: x['name'].lower()):
            html += f'<li><a href="{item["url"]}" target="_blank">{item["name"]} — Click here</a></li>\n'
            
        html += """
            </ul>
        </div>
        """
        
    html += "</div>"
    return html

try:
    html_2nd = process_folder("2nd year", "notes-2nd")
    html_3rd = process_folder("3rd year", "notes-3rd")
    html_pres = process_folder("presentation", "notes-pres")

    with open(INTENTS_PATH, 'r', encoding='utf-8') as f:
        intents_data = json.load(f)

    mappings = {
        "question_papers_2nd_year": {
            "html": html_2nd,
            "patterns": ["Question Papers and Notes - 2nd Year", "Question Papers - 2nd Year", "Second Year - AC", "Second Year - VLSI", "Second Year - DSP", "Second Year - CS", "Second Year - Question Papers", "Second Year - Overall", "give me second year materials", "2nd year notes overall"]
        },
        "question_papers_3rd_year": {
            "html": html_3rd,
            "patterns": ["Question Papers and Notes - 3rd Year", "Question Papers - 3rd Year", "Third Year - DSP", "Third Year - ME", "Third Year - VLSI", "Third Year - Question Papers", "Third Year - Overall", "give me third year materials", "3rd year notes overall"]
        },
        "presentation_files_intent": {
            "html": html_pres,
            "patterns": ["Presentations - Major Project", "Presentations - Technical Seminar", "Presentations - Overall", "Presentation - Abstracts", "Presentation - Reports", "Presentation - Seminars", "Presentation - Overall", "presentation materials", "give me presentation files"]
        }
    }

    for intent_tag, info in mappings.items():
        html_content = info["html"]
        if not html_content: continue
        
        found = False
        for intent in intents_data['intents']:
            if intent['tag'] == intent_tag:
                found = True
                # Merge patterns
                current_patterns = set(intent['patterns'])
                current_patterns.update(info["patterns"])
                intent['patterns'] = list(current_patterns)
                intent['responses'] = [html_content]
                break
        
        if not found:
            intents_data['intents'].append({
                'tag': intent_tag,
                'patterns': info["patterns"],
                'responses': [html_content]
            })

    with open(INTENTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(intents_data, f, indent=4)

    print("Notes synced and intents.json updated successfully.")
except Exception as e:
    print(f"Error occurred: {e}")
