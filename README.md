# Sentiment Analysis of Amazon Product Reviews

## Project Overview
This project aims to classify Amazon product reviews into positive and negative sentiments using natural language processing (NLP) techniques and machine learning models. The dataset used is the Amazon Fine Food Reviews dataset.

## Dataset
- **Source:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Description:** This dataset contains over 500,000 food reviews from Amazon.

## Project Structure
- **Data Preprocessing:** Cleaning and preparing text data for analysis.
- **Exploratory Data Analysis (EDA):** Understanding the dataset and visualizing key patterns.
- **Model Building:** Training and evaluating different machine learning models for sentiment classification.
- **Model Evaluation:** Comparing the performance of different models using various metrics.
- **Optional Deployment:** Building a web application using Streamlit for real-time sentiment analysis.

## Files
- `preprocessing.py`: Code for data preprocessing.
- `eda.py`: Code for exploratory data analysis.
- `model_building.py`: Code for training and evaluating machine learning models.
- `deployment.py`: Code for deploying the model using Streamlit.
- `README.md`: This file.

## Requirements
- Python 3.x
- Libraries: pandas, numpy, re, nltk, scikit-learn, matplotlib, seaborn, wordcloud, streamlit

## How to Run
1. **Data Preprocessing:** Run `preprocessing.py` to preprocess the data.
   ```bash
   python preprocessing.py



### Conclusion for the Project

The sentiment analysis project on Amazon product reviews successfully demonstrated the application of natural language processing and machine learning techniques to classify customer reviews into positive and negative sentiments. Three models—Logistic Regression, Random Forest, and SVM—were built and evaluated. Among these, The Random Forest model performed the best, achieving an accuracy of 0.75, precision of 1.0, recall of 0.5, F1-score of 0.67, and AUC of 0.75. 

The insights gained from this analysis can help businesses understand customer feedback, improve products and services, and make informed decisions. Future work can focus on hyperparameter tuning, incorporating additional features, using model ensembles, and exploring advanced deep learning models to further enhance the performance of sentiment classification.
