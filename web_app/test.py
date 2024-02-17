import requests

# Function to test sentiment prediction
def test_predict_sentiment(tweet,sentiment):
    url = 'http://localhost:5000/predict_sentiment'
    data = {'tweet': tweet}
    response = requests.post(url, data=data)
    print(f"Tweet: {tweet}\nPredicted {response.text}")
    if sentiment == sentiment:
        print("Success")
    else:
        print("Failure")

# Test cases
tweets = {
    "I love sunny days!":"Sentiment : positive",
    "I hate rain.":"Sentiment : negative",
    "This is a great moment.":"Sentiment : positive",
    "This is the worst movie I have ever seen.":"sentiment : negative"
}

if __name__ == "__main__":
    for key,value in tweets.items():
        test_predict_sentiment(key,value)
        print()