# 🏎️ F1 DNF Classification

A Streamlit web application that predicts whether a Formula 1 driver will finish a race or DNF (Did Not Finish) using machine learning.

## 📊 Overview

This project analyzes historical F1 race data to predict DNF outcomes. The interactive dashboard allows you to:
- Load and clean F1 race data from Kaggle
- Train multiple ML models (Random Forest, Gradient Boosting, Logistic Regression, Decision Tree)
- Compare model performance with detailed metrics
- Make real-time predictions for race outcomes

## ✨ Features

- **Multi-Model Comparison** - Train and compare 4 different ML algorithms
- **Feature Engineering** - Create interaction and polynomial features
- **Class Imbalance Handling** - SMOTE, ADASYN, and other resampling techniques
- **Hyperparameter Tuning** - Optimize models using GridSearchCV and RandomizedSearchCV
- **Feature Importance Analysis** - Understand which factors influence DNF predictions
- **Live Predictions** - Interactive tool for real-time DNF probability estimation
- **Data Visualization** - Comprehensive charts and plots for data exploration

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

2. **Run the app:**
   ```bash
   streamlit run Task_2_3_streamlit.py
   ```

3. **Open browser:**
   Navigate to `http://localhost:8501`

## � How to Use

Follow this recommended workflow:

1. **Data Loading** - Click "Load F1 DNF Dataset from Kaggle" to load the data
2. **Data Cleaning** - Click "Clean All Data" to prepare the dataset
3. **Visualization** - Explore feature correlations and distributions
4. **Model Training** - Apply target encoding and train the baseline model
5. **Model Comparison** - Train multiple models and compare their performance
6. **Live Prediction** - Use the interactive tool to predict DNF probability

### Advanced Options

- **Feature Engineering** - Create new features for better predictions
- **Class Imbalance** - Apply resampling techniques for imbalanced data
- **Hyperparameter Tuning** - Fine-tune model parameters
- **Learning Curves** - Diagnose model performance issues

---

*Built with Streamlit • Powered by Vatsal Jain*
