# Sonar Rock vs Mine Classification using Machine Learning in Python
## 📌 Project Overview
This project focuses on building a machine learning model that can accurately classify sonar signals to determine whether an object detected underwater is a Rock or a Mine. Using the Sonar Mines vs Rocks dataset from Kaggle, the project applies data preprocessing, exploratory analysis, and Logistic Regression to understand and predict object types based on 60 numerical sonar features.
The workflow includes splitting the dataset into training and testing sets, evaluating model accuracy, and visualizing performance using metrics such as Accuracy-Score. A simple predictive system is also developed to classify new sonar readings, enabling practical, real-world use.
By integrating data analysis, model training, evaluation, and deployment-ready components, this project demonstrates the complete end-to-end process of building a supervised machine learning classifier.


## 🎯 Objectives
#### Rock vs Mine Classification – Build a machine learning model to classify sonar signal readings and determine whether the detected object is a Rock or a Mine.
#### Data Exploration & Preprocessing – Analyze the Kaggle sonar dataset, clean the data, and prepare it for model training.
#### Model Development - Train classification algorithm - Logistic Regression
#### Performance Evaluation – Evaluate models using metrics like Accuracy-Score, confusion matrix
#### Predictive System Creation – Develop a simple prediction system that takes any input instance (60 sonar features) and outputs whether it is a Mine (M) or a Rock (R).


## 📁 Dataset
#### Name: Sonar, Mines vs Rocks Dataset
#### Source: https://www.kaggle.com/datasets/rupakroy/sonarcsv
#### Shape: 208 samples × 60 features
#### Target labels:
R — Rock
M — Mine


## ⚙️ Methodology
#### Data Preprocessing:
Loaded the Kaggle Sonar dataset, checked the dataset shape, ensured no missing values, and separated features (X) and labels (y).

#### Exploratory Data Analysis (EDA):
Performed initial dataset inspection, visualized class distribution, and reviewed feature ranges to understand signal behavior.

#### Train-Test Split:
Split the data into training and testing sets to evaluate model generalization performance.

#### Model Development (Logistic Regression):
Trained a Logistic Regression classifier on the sonar features to distinguish between Rock (R) and Mine (M).

#### Model Evaluation:
Assessed model performance using Accuracy-Score.
Calculated accuracy for both the training dataset (83.42%) and testing dataset (76.20%).
Predictive System Creation:
Developed a simple prediction system that takes 60 sonar feature values as input and outputs whether the object is a Mine or a Rock.


## 🚀 Results & Insights
#### Model Accuracy:
Logistic Regression achieved 83.42% accuracy on the training dataset and 76.20% accuracy on the testing dataset, showing good generalization.
#### Prediction System:
A custom predictive system was successfully developed to classify any new sonar instance (with 60 features) as either Mine or Rock, enabling real-time detection.
#### Model Reliability:
The consistent accuracy gap between training and testing data indicates the model is neither heavily overfitting nor underfitting—making it dependable for practical use.
#### Key Insight:
The sonar frequency patterns contain enough distinguishable information for a simple linear model (Logistic Regression) to perform reasonably well in classifying underwater objects.


## 🛠 Tech Stack
##### Python: Main programming language used.
##### NumPy: For numerical calculations and array operations.
##### Pandas: For loading, cleaning, and exploring the dataset.
##### Scikit-learn: For building, training, and evaluating machine learning models.
##### Matplotlib / Seaborn: For creating visualizations like confusion matrix and accuracy plots.












