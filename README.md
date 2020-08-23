# Sentiment-Analysis-using-BERT

***** New August 23th, 2020*****

## Introduction

In this project, we will introduce two BERT fine-tuning methods for the sentiment analysis problem for
Vietnamese comments, a method proposed by the BERT authors using only the [CLS] token as the inputs for an attached
feed-forward neural network, a method we have proposed, in which all output vectors are used as inputs for other
classification models. Experimental results on two datasets show that models using BERT is outperforming than other
models. In particular, in both results, our method always produces a model with better performance than the BERTbase method.

## Getting Started

When I work my project, we are working project on Google Colab.

Code structure:

- BERT-base: We use Pretrained BERT model
- BERT-embedding-CNN: We use finetuning BERT combine TextCNN or RCNN
- BERT-embedding-LSTM: We use finetuning BERT combine LSTM
- Data: Data for experiment
- Test: We use other embedding model combine neural network.
- Machine Learning Model: We use finetuning BERT combine machine learning algorithm

### Run

Using google colab to training finetuning BERT combine neural networks.
After, using file predict_text to test with real data.

## Data

*Data description after preprocessing in Data folder*

| Dataset | Train | Test | Totally |
| --- | --- | --- | --- |
| ntc-sv | 20,493 Pos & 20,267 Neg | 5,000 Pos & 5,000 Neg | 50,760 |
| vreview | 22,979 Pos & 19,537 Neg | 8,301 Pos & 6,795 Neg | 57,612 |

*Statistics of words include in the comments*

| | vreview | ntc-sv |
| --- | --- | --- |
| Mean | 55.45 | 86.57 |
| Stdn | 63.75 | 77.41 |
| Min | 1 | 1 |
| 25% | 14 | 37 |
| 50% | 32 | 65 |
| 75% | 76 | 111 |
| Max | 435 | 1,501 |

## Comparison of method

In this section, we will compare several models, which
can be used for sentiment analysis for Vietnamese:

- ***SVM / Boosting***: There are two basic machine
learning algorithms, used much before deep learning
algorithms prevailed, with SVM we use features
from n-grams, n in the range [1,5]. Our Boosting
algorithm will use XGBoosting with a deep of 15.

- ***FastText + LSTM / TextCNN / RCNN***: We choose
three algorithms to combine with word embedding
model FastText is similar to 3 models associated
with BERT, this will help us more accurately evaluate the results that the model of we archive. The
pretrained FastText used has been trained on Vietnamese dataset.

- ***GloVE + LSTM / TextCNN / RCNN***: We will use glove-python6
library to train word embedding. The reason for this is because we don’t have GloVE
pre-trains with the same dimensions as the FastText pre-train. Therefore, it will ensure the fairness for
all. Models associated with GloVE will be similar to models combine with FastText.

## Experiment

The results of comparing the models with the two datasets are shown in the table below. Measure is F1-score.

*Result of our model on NTC-SV Dataset compared to other models*

| Model | Precision(%) | Recall(%) | F1(%) |
| --- | --- | --- | ---|
| SVM | 89.23 | 92.52 | 90.84 |
| XGBoost | 88.76 | 90.58 | 89.63 |
| FastText + TextCNN | 67.9 | 89.1 | 77.1 |
| FastText + LSTM | 88.5 | 89.7 | 89.1 |
| FastText + RCNN | 89.2 | 91.7 | 90.4 |
| Glove + TextCNN | 69.7 | 87.7 | 77.7 |
| Glove + LSTM | 88.7 | 91.8 | 89.8 |
| Glove + RCNN | 85.8 | 85.8 | 90.7 |
| BERT-base | 88.13 | **94.02** | 90.9 |
| BERT-LSTM | **89.78** | 92.08 | 90.91 |
| BERT-TextCNN | 88.85 | 93.14 | 90.94 |
| BERT-RCNN | 88.76 | 93.68 | **91.15** |

*Result of our model on Vreview dataset compared to other models*

| Model | Precision(%) | Recall(%) | F1(%) |
| ---| --- | --- | --- |
| SVM | 86.26 | 86.9 | 86.5 |
| XGBoost | 87.69 | 88.45 | 88.07 |
| FastText + TextCNN | 61.8 | **94** | 74.6 |
| FastText + LSTM | 88.5 | 86.4 | 87.5 |
| FastText + RCNN | 84.5 | 89.8 | 87.1 |
| Glove + TextCNN | 62.6 | 93 | 74.8 |
| Glove + LSTM | 85.8 | 85.8 | 85.8 |
| Glove + RCNN | 84.0 | 88.6 | 86.2 |
| BERT-base | 86.08 | 88.44 | 87.2 |
| BERT-LSTM | 85.25 | 89.9 | 87.5 |
| BERT-TextCNN | **90.9** | 85.2 | 87.98 |
| BERT-RCNN | 87.08 | 89.38 | **88.22** |

## Team
### Members

```
Nguyen Quoc Thai

Nguyen Thoai Linh
```

### Mentor

```
Quoc Hung Ngo

Hoang Ngoc Luong
```

University of Information Technology

Vietnam National University - HCM City

Ho Chi Minh City, Viet Nam

## Paper

Link paper [GitHub](https://github.com/16521716/Sentiment-Analysis-using-BERT/blob/master/Fine-Tuning%20BERT%20on%20Sentiment%20Analysis%20for%20Vietnamese.pdf).
