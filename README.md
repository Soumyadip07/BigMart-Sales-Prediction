# **BigMart Sales Prediction**

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)

## **Project Overview** 📊
This project focuses on predicting sales of BigMart products using machine learning models. The aim is to analyze historical sales data, identify patterns, and build a robust prediction model to forecast future sales.

---

## **Table of Contents**
1. [Problem Statement](#problem-statement)
2. [Dataset Information](#dataset-information)
3. [Approach](#approach)
4. [Technologies Used](#technologies-used)
5. [Setup Instructions](#setup-instructions)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Contact](#contact)

---

## **Problem Statement** 🔍
The goal is to predict the sales of products in various BigMart stores based on historical sales data. Factors like product attributes, store type, and location impact the final sales, which this model aims to uncover.

---

## **Dataset Information** 📂
- **Source:** BigMart Sales Dataset.
- **Training Dataset:** `train_dataset.csv`
- **Test Dataset:** `test_dataset.csv`
- **Columns:**
  - `Item_Weight`
  - `Item_Fat_Content`
  - `Item_Visibility`
  - `Item_Type`
  - `Item_MRP`
  - `Outlet_Identifier`
  - `Outlet_Establishment_Year`
  - `Outlet_Size`
  - `Outlet_Location_Type`
  - `Outlet_Type`
  - `Item_Outlet_Sales` (Target Variable for Training)

---

## **Approach** 🛠️
The following steps were followed in building the predictive model:
1. **Data Cleaning & Transformation**
   - Handle missing values
   - Standardize column names and formats
2. **Exploratory Data Analysis**
   - Visualize relationships between features.
   - Visualize the relationship between features and Target Variable
4. **Feature Engineering**
   - Encoding categorical features using techniques like Label Encoding and One-Hot Encoding.
5. **Model Building**
   - Used machine learning model approache such as:
     - Linear Regression
     - Random Forest[Tuned]
     - XGBoost[Tuned]
     - CatBoost
     - Ensemble Model[Linear Regression, XGBoost, GBM]
     - Random Forest Regressor
     - Gradient Boosting Regressor
   - Used deep learning model approache such as:
     - Custom Feedforward Neural Network
     - Custom CNN [1D Convolutional Neural Network for learning patterns between features]
     - LSTM
     - Ensemble of CNN + LSTM
6. **Evaluation**
   - Measure performance using metrics like MAE,MSE, RMSE and R-squared.

---

## **Technologies Used** 🚀
- Python
- Libraries: Pandas, NumPy, Matplotlib, Scikit-Learn, Tensorflow, XGBoost.
- Jupyter Notebook

---

## **Setup Instructions** ⚙️
1. Clone the repository:
   ```bash
   git clone https://github.com/Soumyadip07/BigMart-Sales-Prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd BigMart-Sales-Prediction
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Open Jupyter Notebook to execute the project files.

---

## **Results** 📈
- Models like **Gradient Boosting Regressor** is ML and **1D CNN** in DL provided high accuracy with low RMSE.
- Comparative analysis and results can be found in `FinalPredictionModel.ipynb`.

---

## **Future Improvements** 🚀
- Perform hyperparameter tuning for further optimization.
- Add more robust feature engineering techniques.

---

## **Contact** 📧
For any queries or collaboration, feel free to reach out:

**SOUMYADIP TIKADER**  
[Email](soumyadiptikader@gmail.com) 

[LinkedIn](https://www.linkedin.com/in/soumyadip-tikader-605393188/)

---
