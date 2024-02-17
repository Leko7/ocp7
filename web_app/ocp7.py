from flask import Flask, request, render_template_string

app = Flask(__name__)

import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from torch import nn

# HTML template for the form
FORM_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Predict Tweet Sentiment</title>
</head>
<body>
    <h1>Enter a tweet</h1>
    <form action="/predict_sentiment" method="post">
        <input type="text" name="tweet" required>
        <input type="submit" value="Predict sentiment">
    </form>
</body>
</html>
"""

# Route for the form
@app.route('/')
def home():
    return render_template_string(FORM_PAGE)

@app.route('/predict_sentiment', methods=['POST'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet'] 
        sentiment = predict_sentiment(tweet)
        return sentiment
    return 'Only POST method is accepted.'

def predict_sentiment(text):
    # Encode the text
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    # Set input_ids and attention_mask
    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    # Get the output
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    # Display the text and associated sentiment prediction
    #print(f'Tweet: {text}')
    #print(f'Sentiment  : {class_names[prediction]}')
    return (f'Sentiment  : {class_names[prediction]}')


# Set GPU
device = torch.device("cpu")

# Define class names, to denote a negative or positive sentiment
class_names = ['negative', 'positive']

# Set the model name
MODEL_NAME = 'bert-base-cased'

# Build a BERT based tokenizer
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# Set max len of tweets as 150 tokens
MAX_LEN = 150

# Build the Sentiment Classifier class
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        pooled_output = bert_output.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# Instantiate the model and move to classifier
model = SentimentClassifier(len(class_names))
model = model.to(device)

# Load the best state of the model
model_path = 'best_model_state.bin'
model.load_state_dict(torch.load(model_path, map_location='cpu'))

if __name__ == '__main__':
    app.run(debug=True)