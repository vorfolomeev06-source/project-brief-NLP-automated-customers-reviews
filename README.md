![logo_ironhack_blue](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)


# NLP Automated Customer Reviews

**Ivan Vorfolomeev | Ironhack AI Engineering**

## About the project

This project is about analyzing Amazon customer reviews using NLP and machine learning.

I worked on four main tasks: sentiment analysis, product clustering, review summarization, and deployment.

My goal was to build a complete workflow, starting with raw customer reviews and finishing with a working application where users can test the sentiment classifier.

I completed all four tasks in one Google Colab notebook, with separate sections for each part.

## Dataset

The dataset is **Datafiniti Amazon Consumer Reviews**, with around 34,000 product reviews.

The main columns I worked with were review text, ratings, product names, and categories.

For sentiment analysis, I converted the ratings into three groups:

- 1–2 stars: Negative
- 3 stars: Neutral
- 4–5 stars: Positive

One of the first problems I noticed was the class imbalance. Around 93% of reviews were positive, which made negative and neutral reviews much harder to predict.

## Task 1 — Sentiment Analysis

I started with TF-IDF to convert review text into numerical features.

Logistic Regression was my baseline model. After that, I compared it with Multinomial Naive Bayes and LinearSVC.

One interesting result was that Naive Bayes achieved around 93% accuracy, but it was predicting positive almost all the time.

This showed me that accuracy alone was not enough for this dataset.

I selected LinearSVC because it performed better across the minority classes.

I also experimented with the TF-IDF settings, including bigrams, feature limits, and filtering rare terms.

### Sentiment results

| Metric | LinearSVC v2 |
|---|---|
| Accuracy | 92.2% |
| Macro F1 | 0.55 |
| Negative F1 | 0.41 |
| Neutral F1 | 0.27 |
| Positive F1 | 0.96 |

The biggest lesson from this task was understanding why a model with high accuracy can still perform badly on minority classes.

For the deployed application, I also tested removing common English stop words. That version reached **91.97% accuracy and 0.53 Macro F1**. It worked better on some manually tested short reviews, although its overall Macro F1 was slightly lower.

## Task 2 — Product Clustering

For clustering, I combined product names, categories, and review text into one input.

Then I converted the text into TF-IDF features and applied **KMeans with 5 clusters**.

KMeans grouped the reviews automatically, but the cluster numbers did not explain what the groups represented.

To understand the results, I checked the top 10 keywords and random reviews from each cluster.

After reviewing them, I gave the clusters these names:

1. Fire tablets
2. Echo / SmartHome
3. Kindle / E-readers
4. Fire TV / Streaming devices
5. Fire HD 8 tablets

Some product groups were quite similar, especially the tablet categories, so interpreting the clusters was not always straightforward.

This task helped me understand that clustering is not only about running a model. The results also need to be checked and interpreted.

## Task 3 — Review Summarization

For this part, I worked with **DistilBART**, a pretrained summarization model from Hugging Face.

Model: `sshleifer/distilbart-cnn-12-6`

I selected around 20 reviews from each cluster, combined them into one text, and generated a short summary.

I also experimented with different input instructions and summary lengths.

I did not fine-tune DistilBART. I used the pretrained model for inference.

Some summaries gave a useful general opinion about the products. However, others focused too much on a single review or specific complaint.

I found this part interesting because the generated text was not always equally representative of all the input reviews.

## Task 4 — Deployment

The final step was creating a working sentiment application using Hugging Face Spaces.

The user can enter a product review, click the prediction button, and receive one of three labels: Positive, Neutral, or Negative.

The application follows this process:

**Review text → TF-IDF → LinearSVC → Sentiment prediction**

### Live application

[Try the Amazon Review Sentiment App](https://huggingface.co/spaces/vorfolomeev06/amazon-review-sentiment)

Getting the application to work outside Google Colab was one of the more challenging parts of the project, especially exporting the model and keeping the text preprocessing consistent.

## How to Run

1. Download the notebook from this repository.
2. Open it in Google Colab.
3. Get the Datafiniti Amazon Consumer Reviews dataset from Kaggle.
4. Upload the dataset to Google Drive and update its path in the notebook.
5. Install the required libraries and run the notebook sections in order.

The dataset is not included in the repository because of its size.

The deployed application can also be tested directly through the link above.

## Project Structure

- `NLP_Automated_Customer_Reviews_Project.ipynb` — main notebook containing the four tasks
- `README.md` — project description and results
- `.gitignore` — excludes large datasets and unnecessary files

## Limitations and Future Improvements

The main limitations were:

- Strong class imbalance
- Lower F1 scores for negative and neutral reviews
- Difficulty understanding short or mixed reviews
- Some overlapping product clusters
- Summaries that occasionally focused too much on individual reviews

With more time, I would try balancing the sentiment data, testing a transformer-based sentiment classifier, and improving how reviews are selected for summarization.

## What I Learned

The most important thing I learned was that choosing a model is not only about getting the highest accuracy.

I also learned how important preprocessing is, how to interpret clustering results, and how different NLP methods can be combined into one project.

Deploying the application was another useful experience because it allowed me to test the model with new reviews outside the notebook.

## Technologies and References

- Python, pandas, scikit-learn
- TF-IDF, Logistic Regression, Multinomial Naive Bayes, LinearSVC
- KMeans clustering
- Hugging Face Transformers and DistilBART
- Hugging Face Spaces
- Dataset: Datafiniti Amazon Consumer Reviews (Kaggle)
