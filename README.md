# Sentiment-Analysis-using-BERT

## Introduction

In this project, we will introduce two BERT fine-tuning methods for the sentiment analysis problem for
Vietnamese comments, a method proposed by the BERT authors using only the [CLS] token as the inputs for an attached
feed-forward neural network, a method we have proposed, in which all output vectors are used as inputs for other
classification models. Experimental results on two datasets show that models using BERT is outperforming than other
models. In particular, in both results, our method always produces a model with better performance than the BERTbase method.

## Data

Data description after preprocessing in Data folder

| Command | Description |
| --- | --- |
| Dataset | Train | Test | Totally |
| --- | --- | --- | --- |
| ntc-sv | 20,493 Pos & 20,267 Neg | 5,000 Pos & 5,000 Neg | 50,760 |
| vreview | 22,979 Pos & 19,537 Neg | 8,301 Pos & 6,795 Neg | 57,612 |
