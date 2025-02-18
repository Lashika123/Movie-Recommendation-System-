# Movie Recommendation System - Data Science & ML Analysis

## Project Overview
This project focuses on analyzing movie ratings data and building a recommendation system using **Exploratory Data Analysis (EDA)** and **Machine Learning techniques**. The dataset includes user ratings for various movies, allowing us to study user behavior, preferences, and trends in movie recommendations.

By leveraging **data preprocessing**, **visualizations**, and **collaborative filtering methods**, this project aims to provide personalized movie recommendations.

## Features
- **Data Loading & Cleaning**: Handling missing values, duplicates, and inconsistencies.
- **Exploratory Data Analysis (EDA)**: Understanding user-movie interactions.
- **Statistical Insights**: Finding popular movies and user trends.
- **Recommendation Algorithms**: Implementing collaborative filtering and content-based filtering.
- **Performance Metrics**: Evaluating model accuracy using RMSE and precision-recall.

## Installation & Requirements
Ensure you have Python installed along with the required dependencies:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn surprise
```

## Code Explanation
### 1. Load and Explore Dataset
```python
import pandas as pd
# Load datasets
movies = pd.read_csv("Movies.csv")
ratings = pd.read_csv("Ratings.csv")
print(movies.head())
print(ratings.info())
```
This step loads the dataset and provides an overview of movie and rating records.

### 2. Exploratory Data Analysis (EDA)
```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(ratings['rating'], bins=10, kde=True)
plt.title("Distribution of Movie Ratings")
plt.show()
```
This visualization helps us understand the distribution of ratings given by users.

### 3. Building a Recommendation System
```python
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)
rmse(predictions)
```
This code builds a **Singular Value Decomposition (SVD) recommendation model**, trains it on the dataset, and evaluates its accuracy.

## How to Use
1. Run the notebook in **Jupyter Notebook** or Google Colab.
2. Explore different movie ratings and trends.
3. Train and test the recommendation model.
4. Modify the model parameters to improve recommendations.



