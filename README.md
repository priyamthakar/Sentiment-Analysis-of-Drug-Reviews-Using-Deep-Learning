# Sentiment Analysis of Drug Reviews Using Deep Learning
## Overview
This project aims to analyze drug reviews by applying sentiment analysis techniques to understand the sentiments behind the reviews. We utilize a variety of Python libraries including Pandas, NumPy, Matplotlib, Seaborn, SKLearn, NLTK, and TensorFlow to preprocess and analyze the dataset from the UCI Machine Learning Repository. The project culminates in the development of a Long Short-Term Memory (LSTM) model to predict sentiments based on the review texts. 

## Data
The dataset, obtained from the UCI ML Drug Review dataset, consists of drug reviews including the unique ID, drug name, condition, review text, rating, date, and useful count. Initial preprocessing involves cleaning missing values, converting the date column to datetime format, and exploring the distribution of ratings and the frequency of reviews for different drugs and conditions. https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

## Analysis
The analysis begins with exploring basic operations using Pandas, NumPy, Matplotlib, and Seaborn to understand the dataset's structure, including:
- Distribution of ratings
- Most reviewed drugs
- Conditions leading to most reviews
- Relationship between review usefulness and ratings
- Time series analysis of reviews

## Text Processing & LSTM Model
Text data from reviews undergo preprocessing to lowercasing, remove HTML tags, URLs, punctuation, numbers, and extra whitespace. This cleaned text is then tokenized and padded to prepare for input into an LSTM model.

The LSTM model is structured with an Embedding layer, an LSTM layer, and a Dense output layer, utilizing the Sparse Categorical Crossentropy loss function for multi-class classification of review sentiments. Training involves a categorical conversion of rating scores and splitting the dataset into training and validation sets.

## Model Training & Evaluation 
The model is trained with the aim to predict the sentiment (rating) based on the review text. Despite the complexity of sentiment analysis, the LSTM model shows promising results, capturing the nuances of sentiment in drug reviews.

## Searching and Predicting Sentiments for Specific Drugs
A custom function allows for searching reviews of a specific drug and predicting sentiments using the trained model. This demonstrates the model's application in real-world scenarios, such as understanding patient sentiment towards specific medications.

## Conclusion
This project demonstrates the application of deep learning in analyzing and predicting sentiments from drug reviews. The LSTM model provides valuable insights into patient sentiments, which can be useful for pharmaceutical companies, healthcare providers, and patients looking to understand the efficacy and satisfaction levels of various medications.

## Technologies Used
- Python 3.11.5
- Pandas, NumPy, Matplotlib, Seaborn
- SKLearn for machine learning operations
- NLTK for natural language processing
- TensorFlow and Keras for building and training the LSTM model

Note: This project is for educational purposes and the dataset used is from the UCI Machine Learning Repository. The model accuracy and performance may vary based on the dataset's nature and the chosen parameters.
