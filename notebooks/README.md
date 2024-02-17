# ocp7 - Notebooks
This repository contains the notebooks with all the info about data processing, model selection, model training.

I tried three main approaches (with small variants in each):

- a tf-idf method with a logistic regression
- a NN approach with embedding ans LSTMs (using Tensorflow)
- a NN approach with transformers (using Pytorch and a BERT model from the Transformers library)

For each of these approaches, I wrote 2 scripts, one for training the model, and one for performing a new inference about the sentiment of a tweet.

I also wrote a notebook for the whole model comparison/selection and visualisation process.


# Files Description

- infer_bert_transformers.ipynb : inference part of a NN approach with transformers (using Pytorch and a BERT model from the Transformers library)
- infer_log_reg_tf_idf_baseline.ipynb : inference part of a tf-idf method with a logistic regression
- infer_nn_embd_lstms.ipynb : inference part of a NN approach with embedding ans LSTMs (using Tensorflow)
- model_comparison.ipynb : interactive notebook for the whole model comparison/selection and visualisation process
- train_bert_transformers.ipynb : training part of a NN approach with transformers (using Pytorch and a BERT model from the Transformers library)
- train_log_reg_tf_idf_baseline.ipynb : training part of a tf-idf method with a logistic regression
- train_nn_embd_lstms.ipynb : training part of a NN approach with embedding ans LSTMs (using Tensorflow)
