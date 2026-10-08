![logo_ironhack_blue](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Project | NLP Automated Customer Reviews


This project builds an NLP pipeline for analyzing Amazon customer reviews.

The project includes sentiment classification, product category clustering, review summarization with a pretrained generative model, and deployment of the sentiment classifier as a public web application.

## Dataset

I used the Datafiniti Amazon Consumer Reviews dataset.

The main file contains more than 34,000 Amazon product reviews.

For the project I mainly used:

- review text
- star rating
- product name
- product categories

For sentiment analysis, ratings were mapped into three classes:

- 1–2 stars → Negative
- 3 stars → Neutral
- 4–5 stars → Positive

The dataset was highly imbalanced, with most reviews belonging to the positive class.

## Task 1 — Sentiment Analysis

I used TF-IDF to convert review text into numerical features.

I compared:

- Logistic Regression
- Multinomial Naive Bayes
- LinearSVC

LinearSVC was selected because it provided better performance across the minority classes.

Final results:

- Accuracy: 92.2%
- Macro F1: 0.55
- Negative F1: 0.41
- Neutral F1: 0.27
- Positive F1: 0.96

The main limitation was class imbalance, which made negative and neutral reviews more difficult to classify.

## Task 2 — Product Category Clustering

For clustering, I combined:

- product categories
- product names
- review text

The text was transformed using TF-IDF and clustered using KMeans with 5 clusters.

The final clusters were:

1. Fire tablets
2. Echo / SmartHome
3. Kindle / E-readers
4. Fire TV / Streaming devices
5. Fire HD 8 tablets

Cluster names were assigned after inspecting the top keywords and sample reviews from each cluster.

## Task 3 — Review Summarization

I used the pretrained DistilBART model:

`sshleifer/distilbart-cnn-12-6`

For each cluster, I sampled reviews, combined them into one input text, and generated a short summary.

I also experimented with different prompt variants and summary lengths.

DistilBART was used for inference without fine-tuning.

## Task 4 — Deployment

The sentiment classifier was deployed as a public web application using Hugging Face Spaces.

Users can enter a product review and receive a prediction:

- Positive
- Neutral
- Negative

### Live Demo

https://huggingface.co/spaces/vorfolomeev06/amazon-review-sentiment

## Main Limitations

- Strong class imbalance
- Lower performance for negative and neutral reviews
- TF-IDF does not fully understand semantic context
- Some product clusters overlap
- Generated summaries may focus too strongly on individual sampled reviews

## Future Improvements

Possible improvements include:

- balancing the sentiment dataset
- testing transformer-based sentiment models
- using embeddings for richer text representations
- improving review sampling for summarization
- fine-tuning a generative model for category summaries


https://huggingface.co/spaces/vorfolomeev06/amazon-review-sentiment
