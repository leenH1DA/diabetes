📊 Diabetes Prediction Web App – Project Overview
🎯 Project Goal
The objective of this project is to predict the likelihood of diabetes using a set of health indicators derived from the CDC’s BRFSS 2015 dataset. The project leverages machine learning techniques—particularly the XGBoost classifier—to build a predictive model, and presents the result through an interactive web application built with Streamlit.

The model is trained on a carefully selected subset of features using feature selection methods, enabling efficient and interpretable predictions based on a minimal number of key health indicators.
📁 About the Dataset
🧠 Context
Diabetes is a chronic disease affecting millions globally. It interferes with the body's ability to regulate blood glucose levels due to issues with insulin production or function. Without intervention, diabetes can lead to severe complications including heart disease, kidney failure, and vision loss.

Early detection is critical and predictive tools based on public health surveys can provide low-cost, scalable solutions for identifying at-risk individuals.

📊 Data Source
The data used in this project originates from the Behavioral Risk Factor Surveillance System (BRFSS), one of the largest annual health surveys conducted by the CDC. For this app:

We use the file: diabetes_012_health_indicators_BRFSS2015.csv

It includes 253,680 responses and 21 features

The target variable is Diabetes_012 with 3 classes:

0: No diabetes or only during pregnancy

1: Prediabetes

2: Diabetes

In this project, classes 1 and 2 are merged into a single class to simplify prediction into a binary classification problem:

0: No diabetes

1: Prediabetes or diabetes

🧪 Machine Learning Pipeline
Feature Selection: Based on domain knowledge and statistical methods, the following key features were selected:

General Health (GenHlth)

Physical Health (PhysHlth)

Mental Health (MentHlth)

Difficulty Walking (DiffWalk)

Income

BMI

High Blood Pressure (HighBP)

Data Processing:

Standardization using StandardScaler

Handling class imbalance using SMOTE

Model:

XGBoostClassifier with hyperparameters fine-tuned for performance

Evaluation:

Accuracy score

Classification report (precision, recall, f1-score)

🌐 Web Application
The application was developed using Streamlit, allowing users to:

Input their own health data through sliders and dropdowns

View instant predictions on whether they are at high risk of diabetes

See visual feedback depending on risk type (Type 1 vs. Type 2 based on age)

🎨 The app includes:

Custom styling and theming (CSS)

Dynamic background images and result images

Caching for model loading and training to optimize performance

🔍 Research Questions Explored
Can health survey data predict diabetes accurately?

What are the most predictive health indicators?

Can we reduce the number of questions while maintaining accuracy?

How can we assist public health awareness using predictive technology?

🙏 Acknowledgements
This project builds on a cleaned version of the BRFSS 2015 dataset available on Kaggle. Special thanks to Zidian Xie et al. whose research on machine learning models for diabetes prediction using the 2014 BRFSS inspired this approach.

🚀 Inspiration
Inspired by the work "Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques" by Zidian Xie et al., this project brings the insights from public health research into a usable form for the general public and developers alike.
