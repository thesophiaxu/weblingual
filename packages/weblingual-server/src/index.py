from ner_pipeline import TokenClassificationPipeline

from flask import Flask, escape, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from transformers import pipeline, GPT2Model, AutoTokenizer, GPT2Tokenizer, GPT2ForSequenceClassification, BertModel, AutoModelForTokenClassification, BertTokenizer, BertForTokenClassification
import json
# Environmenal Variables
VERSION = "0.0.1"

import sys
model_loc = sys.argv[1]
if not model_loc:
    model_loc = "dbmdz/bert-large-cased-finetuned-conll03-english"

# Models and Pipelines
#model = GPT2ForTokenClassification.from_pretrained('gpt2-large')
#model_classification = GPT2ForSequenceClassification.from_pretrained('gpt2-large')
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
#model_token_classification = BertForTokenClassification.from_pretrained('bert-base-uncased')
#model_classification = BertForSequenceClassification.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#pipeline_sentiment = pipeline('sentiment-analysis', model=model_classification, tokenizer=tokenizer)
#pipeline_ner = pipeline('ner', model=model_token_classification, tokenizer=tokenizer)
model_ner = AutoModelForTokenClassification.from_pretrained(model_loc).to('cuda')
tokenizer_ner = AutoTokenizer.from_pretrained('bert-large-cased')
pipeline_ner = TokenClassificationPipeline(task="ner", model=model_ner, tokenizer=tokenizer_ner, grouped_entities=True, device=0)
#pipeline_ner = pipeline('ner', grouped_entities=True)

# Handlers
def handle_ner(texts):
    return pipeline_ner(texts)

def handle_sentiment(text):
    #return pipeline_sentiment(text, padding=True)
    pass

@app.route('/')
def root():
    return '''WebLingual server version {}
APIs:
- /tasks/ner POST JSON
    Accepts a JSON object like \{"text": "Paris is the capital of France"\}, where the text field is the text to determine.
    Returns a JSON object like \{"result": ...\}, where result field is the result of the task.
'''.format(VERSION)

@app.route('/tasks/sentiment', methods=['POST'])
def sentiment_analysis():
    text = request.get_json(force = True)['text']
    print(text)
    result = handle_sentiment(text=text)
    return json.dumps(str({"result": result[0]}))

@app.route('/tasks/ner', methods=['POST'])
def ner_task():
    texts = request.get_json(force = True)['texts']
    print(texts)
    result = handle_ner(texts=texts)
    return json.dumps(str({"result": result}))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=22222)