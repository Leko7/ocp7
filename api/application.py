#from analyze_sentiment import dummy
#from analyze_sentiment import analyze_sentiment
from flask import Flask, request, jsonify
app = Flask(__name__)

import re
import string
import nltk
from nltk.corpus import stopwords
from joblib import load
import pickle

nltk.download('wordnet')
nltk.download('punkt')

def cleaning_stopwords(text,stopwords):
    return " ".join([word for word in str(text).split() if word not in stopwords])

def cleaning_punctuations(text, punctuations_list):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

def cleaning_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_URLs(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)

def cleaning_email(data):
    return re.sub('@[^\s]+', ' ', data)

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

def stemming_on_text(data,st):
    text = [st.stem(word) for word in data]
    return data

def lemmatizer_on_text(data,lm):
    text = [lm.lemmatize(word) for word in data]
    return data

def analyze_sentiment_baseline(text):

    # Load vectorizer
    with open('vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    # Load the prediction model
    model = load('baseline.joblib')

    # Enter a new tweet
    new_tweet = text
    new_tweet_original = new_tweet

    ### Converting uppercase to lowercase
    new_tweet=new_tweet.lower()

    ### Removing English stopwords
    #stopwords_list = stopwords.words('english')

    ### Removing stop words
    STOPWORDS = set(stopwords.words('english'))
    new_tweet = cleaning_stopwords(new_tweet, STOPWORDS)

    ### Removing punctuation
    english_punctuations = string.punctuation
    punctuations_list = english_punctuations
    new_tweet= cleaning_punctuations(new_tweet, punctuations_list)

    ### Removing repeated characters
    new_tweet = cleaning_repeating_char(new_tweet)

    ### Removing emails
    new_tweet= cleaning_email(new_tweet)

    ### Removing URLs
    new_tweet = cleaning_URLs(new_tweet)

    ### Removing numbers
    new_tweet = cleaning_numbers(new_tweet)

    ### Tokenization of tweets
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    new_tweet = tokenizer.tokenize(new_tweet)

    ### Stemming
    st = nltk.PorterStemmer()
    new_tweet= stemming_on_text(new_tweet,st)

    ### Lemmatization
    lm = nltk.WordNetLemmatizer()
    new_tweet = lemmatizer_on_text(new_tweet,lm)

    ### Bag of Words Approach: Tf-idf

    ### Formatting needed for Bow approach
    new_tweet = ' '.join(new_tweet)

    ### Vectorize the new tweet
    X = vectorizer.transform([new_tweet])

    ### Get the prediction on the new tweet
    y_pred = model.predict(X)

    ### Get the sentiment
    sentiment = "negative" if y_pred < 0.5 else "positive"

    return sentiment


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data['text']
    sentiment = analyze_sentiment_baseline(text)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
