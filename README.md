# Semantic-Book-Recommender
Built a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio)
![Screenshot 2025-02-28 at 1 01 00 PM](https://github.com/user-attachments/assets/2734b7d3-3a3a-4eb0-b30e-f0ad905a8b9f)

## Overview
This project is a Semantic Book Recommender that uses Natural Language Processing (NLP) techniques to recommend books based on their descriptions and user preferences. It leverages machine learning models to predict the most relevant books.

## Files

- `data-exploration.ipynb`: This Jupyter notebook contains the data exploration steps, including cleaning and preparing the data.
- `gradio-dashboard.py`: A Python script to build an interactive Gradio dashboard for the book recommendation system.
- `books_cleaned.csv`: The cleaned version of the book dataset used for training and predictions.
- `books_with_categories.csv`: Contains book data with category labels.
- `books_with_emotions.csv`: Contains book data enriched with emotional sentiment labels.
- `sentiment-analysis.ipynb`: This Jupyter notebook focuses on sentiment analysis, analyzing the emotional tone of book descriptions to enhance recommendation accuracy.

## Features

- **Data Preprocessing**: The data is cleaned and processed for further analysis.
- **Sentiment Analysis**: Sentiment classification of books based on descriptions.
- **Text Classification**: Classifies books into different categories.
- **Gradio Dashboard**: An interactive UI to interact with the recommender system.

## Technologies Used

- Python
- pandas, numpy
- Scikit-learn
- TensorFlow/Keras (for sentiment analysis and text classification)
- Gradio (for interactive dashboard)
- Kaggle API for dataset download

## Dataset

The dataset used in this project is the [7k Books with Metadata](https://www.kaggle.com/dylanjcastillo/7k-books-with-metadata), which can be downloaded using the following path:

```python
kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")

