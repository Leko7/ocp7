# ocp7
This repository contains all my code for the OpenClassrooms project number 7.
The goal is to create a sentiment analysis model, able to predict if a tweet is associated to a positive or negative sentiment.

I tried three main approaches (with small variants in each):

- a tf-idf method with a logistic regression
- a NN approach with embedding ans LSTMs (using Tensorflow)
- a NN approach with transformers (using Pytorch and a BERT model from the Transformers library)

For each of these approaches, I wrote 2 scripts, one for training the model, and one for performing a new inference about the sentiment of a tweet.

I also wrote a notebook for the whole model comparison/selection and visualisation process.


# Files Description

- model_comparison.ipynb : interactive notebook for the whole model comparison/selection and visualisation process.
- model_
