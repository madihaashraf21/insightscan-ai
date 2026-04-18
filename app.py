from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy

app = Flask(__name__)

# 1. Load Summarization Model (The "Summarizer")
model_name = "Falconsai/text_summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 2. Load NLP Scanner (The "Insight Scanner")
nlp = spacy.load("en_core_web_sm")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        text = request.form['rawtext']
        
        if not text.strip():
            return render_template('index.html', summary="Please enter some text first!", entities=[], original_text="")

        # --- PART A: AI SUMMARIZATION ---
        inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=100, 
            min_length=30, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # --- PART B: INSIGHT SCANNING (NER) ---
        doc = nlp(text)
        
        # Mapping labels to easy English names
        label_map = {
            'PERSON': 'Person',
            'GPE': 'Location',
            'ORG': 'Organization',
            'DATE': 'Date',
            'NORP': 'Nationality/Group',
            'LOC': 'Location',
            'FAC': 'Building/Airport'
        }
        
        entities = []
        for ent in doc.ents:
            if ent.label_ in label_map:
                clean_label = label_map[ent.label_]
                entities.append((ent.text, clean_label))

        unique_entities = list(set(entities))
        
        return render_template('index.html', 
                               summary=summary_text, 
                               entities=unique_entities, 
                               original_text=text)

if __name__ == '__main__':
    app.run(debug=True)