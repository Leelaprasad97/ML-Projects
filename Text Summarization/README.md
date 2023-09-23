# Qualitative Evaluation of Abstractive Text Summarization

## Overview
This project aims to evaluate the performance of Google’s Flan-T5 and Pegasus models for abstractive text summarization on the Kaggle News articles Dataset and Hindi news articles. The study includes a comparative analysis of the two models, emphasizing the quality of generated summaries.


## Table of Contents

1. [Novelty](#novelty)
2. [Introduction](#introduction)
3. [Text Summarization](#text-summarization)
4. [Google’s Flan-T5 Model](#google’s-flan-t5-model)
5. [Pegasus Model](#pegasus-model)
6. [Datasets](#datasets)
7. [Methodology](#methodology)
8. [Results & Discussion](#results--discussion)
9. [Conclusion](#conclusion)
10. [References](#references)

---

## Novelty

- This project leverages LLMs trained models and libraries based on a reference paper [5] to automatically generate concise summaries from lengthy texts.
- It guides users in selecting the most suitable model based on size and training time.
- Compares the Flan-T5 model published in Dec 2022 with the model in the reference paper [5] using a different dataset.
- Expands the project scope to include evaluation on Hindi dataset.

## Introduction

This project conducts a comparative analysis of the Flan-T5 model and PEGASUS algorithm using a Kaggle news article dataset. It evaluates the summarization quality with a focus on generated summaries. The study explores real-world applications and model selection considerations.

## Text Summarization

Text summarization condenses text into concise summaries containing key information. It enhances productivity and comprehension by saving time and enabling quick understanding of core messages. Techniques include extractive and abstractive summarization.

- Extractive Summarization: Selects existing sentences or phrases from the original text to create a summary.
- Abstractive Summarization: Utilizes NLP techniques to generate summaries that may not be composed of original sentences or words.

## Google’s Flan-T5 Model

Flan-T5 is a state-of-the-art language model based on Google's T5 architecture. It is available in different sizes ranging from Flan-small to Flan-XLL with varying weights. The model employs an encoder-decoder structure for processing input text and generating output.

- Architecture: Based on T5 model, it uses encoder-decoder structure with self-attention layers.
- Tokenization: Breaks down input text into smaller units (tokens) for processing.
- Application: Trained on a 750 GB dataset for various tasks including text summarization.

## Pegasus Model

PEGASUS is an abstractive summarization algorithm developed by Google. It is trained on over 5TB of data and uses gap-sentences generation and masked language model for pre-training. The model is designed for abstractive text summarization and does not require additional training.

- Architecture: Sequence-to-sequence model with Transformer encoder-decoder architecture.
- Pre-training Objectives: Gap sentence generation and masked language model.

## Datasets

- Kaggle News Article Dataset: Utilizes train_ds.csv and val.csv with 80,000 articles for evaluation.
- Hindi News Article Dataset: Contains 17500 articles and summaries, preprocessed for model evaluation.

## Methodology

The project employs Rouge-1, Rouge-2, and Rouge-L scores for evaluating summarization quality. It outlines the training process for both Flan-T5 and PEGASUS models and discusses loss functions.

## Results & Discussion

The Flan-T5 model outperforms PEGASUS in all Rouge metrics on the English dataset. However, the performance on the Hindi dataset is subpar, indicating the need for further investigation.

## Conclusion

This project highlights the significance of automatic text summarization. Flan-T5 demonstrates superior performance over PEGASUS on English articles. The performance on Hindi articles suggests possible areas for improvement.

## References

1. [Flan-T5 models](https://arxiv.org/pdf/2210.11416v5.pdf)
2. [T5- Transformer architecture](https://medium.com/analytics-Vidhya)
3. [Pegasus model](https://arxiv.org/pdf/1912.08777v2.pdf)
4. [Kaggle dataset](Kaggle Text summarization data)
5. Qualitative analysis reference: Research paper
