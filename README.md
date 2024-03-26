# Sentiment Analysis of Drug Reviews Using Deep Learning
## Overview
This project aims to analyze drug reviews by applying sentiment analysis techniques to understand the sentiments behind the reviews. We utilize a variety of Python libraries including Pandas, NumPy, Matplotlib, Seaborn, SKLearn, NLTK, and TensorFlow to preprocess and analyze the dataset from the UCI Machine Learning Repository. The project culminates in the development of a Long Short-Term Memory (LSTM) model to predict sentiments based on the review texts. 

## Data
The dataset, obtained from the UCI ML Drug Review dataset, consists of drug reviews including the unique ID, drug name, condition, review text, rating, date, and useful count. Initial preprocessing involves cleaning missing values, converting the date column to datetime format, and exploring the distribution of ratings and the frequency of reviews for different drugs and conditions. https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018

## Analysis
The analysis begins with exploring basic operations using Pandas, NumPy, Matplotlib, and Seaborn to understand the dataset's structure, including:
- Distribution of ratings
- Top 10 Most reviewed drugs
- Top 10 Most common disease conditions
- Ratings vs Useful Counts
- Reviews over time
- Ratings over time
- Useful counts over time
- Evolution of Review Ratings & Usefulness Over Time
- LSTM Model Training, Evaluation & Summary
- Review of drugs & Corresponding ratings
- Example functions to search a drug name & reviews


## Visualization 

**1. Distribution of Ratings:**
The bar chart indicates a strong tendency for users to leave high ratings, with 10s being the most common, followed by 9s and then 1s. The skewed distribution towards higher ratings suggests that users who have had positive experiences with their medications are more likely to share their feedback. Conversely, the significant count of low ratings, particularly 1s, points to a segment of users who had negative experiences, possibly due to ineffectiveness or adverse effects of the drugs. The skewness of the distribution could imply reporting bias or genuine satisfaction among most users.

**2. Evolution of Review Ratings and Usefulness Over Time:**
The dual line graph shows a downward trend in both average ratings and the average useful count. Initially, both metrics start at higher values but decline over time, which might indicate that over the years, either the users became more critical in their assessments, or the quality of reviews has diminished. Alternatively, this could reflect a change in the user base or platform policies affecting the perception of what constitutes a useful review.

**3. Rating vs. Useful Count:**
This scatter plot reveals that higher ratings do not necessarily correlate with higher usefulness counts. The distribution of points is widespread, with some high ratings receiving low usefulness counts and some lower ratings receiving high usefulness counts. This suggests that the perceived usefulness of a review to other users does not depend strictly on the rating given but possibly on the quality of the review content.

**4. Rating Over Time:**
The time-series graph displays a gradual decline in average ratings over a decade. The decline could be due to various factors such as changes in user demographics, evolving expectations, a rise in the number of critical reviews, or actual variations in drug efficacy or side effects profiles becoming more well-known over time.

**5. Reviews Over Time:**
This graph presents the count of reviews through time, showing periodic peaks and troughs. These fluctuations might correspond with external factors such as new drug releases, health scares, seasonal health issues, or the growing or waning popularity of the review platform itself.

**6. Top 10 Conditions:**
The bar chart lists the health conditions most frequently mentioned in reviews. The prominence of birth control, depression, and pain might reflect their high prevalence or public awareness. It could also indicate a high volume of medications available and used for these conditions, leading to a larger body of reviews.

**7. Top 10 Most Reviewed Drugs:**
The bar chart here illustrates which medications are most frequently reviewed. Levonorgestrel and Ethinyl estradiol, common in birth control medications, dominate the chart, hinting at the extensive usage of these drugs. This chart could inform pharmaceutical companies and healthcare providers about the most commonly used medications and the active user engagement in discussing these drugs.

**8. Useful Count Over Time:**
The plot shows the useful count of reviews over time, starting high but declining steadily. This pattern could reflect changes in platform user engagement, evolving definitions of what users consider helpful, or possibly a general trend of user attrition or review fatigue.

The visualizations offer a multi-faceted picture of drug reviews and user engagement over time. The analysis of such data could be invaluable for pharmaceutical companies, healthcare providers, and policymakers in understanding patient experiences, improving drug development, and enhancing healthcare delivery.

## LSTM Model : 

This deep learning project uses a Long Short-Term Memory (LSTM) model, a type of recurrent neural network suitable for sequence prediction problems, to analyze and predict ratings based on drug reviews. Here's a detailed overview of the model's pipeline:

1. **Data Cleaning Function (`clean_text`)**: A function to preprocess the review text is defined. It standardizes the text by converting it to lowercase, removing HTML tags, URLs, punctuation, numbers, and extra whitespace, creating a clean dataset that is more suitable for NLP tasks.

2. **Data Preprocessing**:
    - The `clean_text` function is applied to the `review` column of a DataFrame `df`.
    - Reviews are converted to a list, and the tokenizer from the Keras library is used to convert text data into sequences of integers.
    - The sequences are then padded to a maximum length to ensure consistent input size for the neural network.
    - Ratings are one-hot encoded to transform them into categorical labels suitable for classification.

3. **Data Splitting**: The dataset is split into training and validation sets, allowing the model to learn from one subset of data and be evaluated on another.

4. **Model Architecture**:
    - An `Embedding` layer is used to turn positive integers (indexes) into dense vectors of fixed size, which is a common representation for processing text with neural networks.
    - An `LSTM` layer is included to allow the model to learn from the sequential nature of the text data.
    - A `Dense` layer with a `softmax` activation function is used for multi-class classification, predicting the probability of each rating class.

5. **Model Compilation**: The model is compiled with the `adam` optimizer and `categorical_crossentropy` loss function, appropriate for multi-class classification tasks.

6. **Model Training**: The model is trained on the training set for a fixed number of epochs with batched data.

7. **Performance Evaluation**: The output indicates that as training progresses, the model's accuracy on the training set increases significantly. However, the validation accuracy does not improve as much, and the validation loss increases, suggesting potential overfitting to the training data.

8. **Function for Drug Review Prediction (`search_drug`)**:
    - A function is provided to search for reviews of a specific drug and predict their ratings using the trained model.
    - It preprocesses the text, tokenizes it, pads it, and then uses the model to predict the rating.

9. **Model Summary and Evaluation**:
    - The model summary indicates the structure and parameters of the network, including the number of neurons and trainable parameters.
    - The model is evaluated on the validation set, and the accuracy is printed out, indicating the model's performance on unseen data.

10. **Example of Prediction**:
    - An example function `search_drug_review` demonstrates how to use the model to predict the rating of a specific review text.
    - The prediction process includes text cleaning, tokenizing, padding, predicting with the model, and interpreting the predicted probability as a class label for the rating.


## **Outcomes:**
- A text preprocessing pipeline was successfully established, incorporating best practices for cleaning and standardizing user-generated content.
- An NLP model using an LSTM architecture was built and trained, leveraging the sequential nature of language data to predict user ratings from review texts.
- The model showed increasing accuracy over training epochs on the training data set but struggled to generalize effectively to the validation data, as evidenced by the divergence between training accuracy and validation accuracy.

## **Implications:**
- The data preprocessing steps indicate the importance of clean and normalized data for NLP tasks, which significantly impact the model's performance.
- The model's ability to learn from the training dataset and improve its performance over time suggests that given sufficient data and tuning, LSTMs can model the nuances of language in user reviews.
- The discrepancy between training and validation performance suggests that the model may be overfitting to the training data. This overfitting can lead to poor generalization on unseen data and may necessitate the introduction of regularization techniques or more training data.

## **Challenges and Areas for Improvement:**
- **Model Overfitting**: The increasing validation loss points to overfitting, a common challenge in machine learning. To address this, one could explore regularization methods, dropout layers in the LSTM, or more complex models that might better capture the nuances of language.
- **Data Representation**: The embedding layer uses a fixed-size vector for each word, which does not account for the polysemous nature of language (words having multiple meanings). Future iterations could use pre-trained embeddings like Word2Vec or GloVe, or contextual embeddings from models like BERT.
- **Hyperparameter Tuning**: It appears that there was no hyperparameter tuning process described. Implementing a strategy to find optimal hyperparameters could improve model performance.
- **Review Length**: The model truncates or pads reviews to a fixed size (100 tokens), which might result in loss of context, especially for longer reviews. Adjusting the maximum review length or employing attention mechanisms might yield better results.
- **Class Imbalance**: The dataset might contain an imbalance in the distribution of ratings, which can bias the model. Techniques such as weighted loss functions could help mitigate this.
- **Evaluation Metrics**: While accuracy is used as the evaluation metric, it may not be the best choice given the potential class imbalance. Other metrics like F1-score, Precision, Recall, or a confusion matrix could provide a more nuanced evaluation.

## **Future Directions:**
- **Expand Dataset**: A larger, more diverse dataset could help the model learn a more general representation of the language used in reviews.
- **Advanced Models**: Experimenting with more sophisticated NLP models, such as transformers or incorporating attention mechanisms, could capture long-range dependencies better than LSTMs.
- **Interpretability**: Implementing model interpretability tools like LIME or SHAP could provide insights into which words or phrases are most predictive of ratings, which can be invaluable for understanding user sentiment.

In conclusion, the project demonstrates the potential of LSTM models in processing and predicting outcomes from sequential text data. While the model shows promise, there is room for improvement, particularly in generalizing to new data, which is essential for real-world applications. Future work can address these challenges, refine the model, and leverage more advanced techniques to improve performance and reliability.

## Technologies Used
- Python 3.11.5
- Pandas, NumPy, Matplotlib, Seaborn
- SKLearn for machine learning operations
- NLTK for natural language processing
- TensorFlow and Keras for building and training the LSTM model

Note: This project is for educational purposes and the dataset used is from the UCI Machine Learning Repository. The model accuracy and performance may vary based on the dataset's nature and the chosen parameters.
