# Nm_phase2
Phase-2 Submission Template 

Student Name: Jasmine .R

Register Number: 412723104046

Institution: TAGORE ENGINEERING COLLEGE 

Department: COMPUTER SCIENCE AND ENGINEERING

Date of Submission: 02.05.2025 

1.	Problem Statement 

Diabetes is one of the most common chronic diseases affecting millions of people globally. Many individuals are unaware of their condition until it leads to serious complications such as heart disease, kidney failure, or vision loss. In rural or under-resourced areas, access to medical diagnostics and timely health screenings is limited.

Traditional diagnostic procedures require clinical visits, lab tests, and healthcare professional intervention, which may not always be feasible or accessible for all individuals.







To address this challenge, our project proposes the development of an AI-powered web-based prediction system that allows users to enter their health-related information—such as glucose level, BMI, age, and insulin level—to determine the likelihood of having diabetes. The goal is to provide early risk detection, promote preventive care, and increase awareness among users, especially in remote areas.

This tool can assist in:

●	Raising awareness among individuals at risk.

●	Providing an easy-to-use platform for preliminary risk assessment.

●	Supporting healthcare professionals with a supplementary diagnostic tool.



2.	Project Objectives 

   Early Detection of Diabetes: To predict the likelihood of diabetes using machine learning based on user-input medical data.

  Build a User-Friendly Web Interface: To design and develop a simple, interactive website where users can input their health data and receive instant predictions.

   AI-Powered Prediction Model: To integrate a trained machine learning model capable of delivering accurate predictions using real-world medical datasets.

   Real-Time Diagnosis Support: To offer instant results that help users decide if they need further medical consultation.

   Deploy a Scalable Solution: To deploy the application in production mode with backend (Flask) and optionally integrate Firebase for database support.

   Data Privacy and Accessibility: To ensure that the system is secure, accessible from anywhere, and handles user data responsibly.

 

3.	Flowchart of the Project Workflow 

 

4.	Data Description  

The dataset used in this project contains medical data that can help predict whether a patient is likely to develop diabetes. It includes multiple features related to the health status of individuals. The data is commonly used for classification problems and helps in predicting the likelihood of diabetes based on various health parameters.





Features:

1.	Pregnancies: The number of times the patient has been pregnant. This feature represents the number of pregnancies in the patient's medical history. It can give insight into certain health risks.

2.	Glucose: The blood glucose concentration in the patient’s body. This feature is crucial in diabetes diagnosis as elevated glucose levels are one of the main indicators of diabetes.

3.	Blood Pressure: The patient's blood pressure. High blood pressure can be an indicator of cardiovascular issues, which are often linked to diabetes.

4.	Skin Thickness: The skinfold thickness in mm, measured on the tricep. Higher skinfold thickness can sometimes correlate with obesity, which is a significant risk factor for diabetes.

5.	Insulin: The insulin level in the patient's body. Low or high insulin levels are key indicators of diabetes, especially Type 2 diabetes.

6.	BMI (Body Mass Index): The BMI of the patient, calculated based on weight and height. A higher BMI is often associated with a higher risk of developing diabetes.

7.	Diabetes Pedigree Function: A function that represents the genetic influence on the likelihood of diabetes. This feature is used to account for the genetic predisposition to diabetes.

8.	Age: The age of the patient. Age is a critical factor, as the risk of developing diabetes increases with age.







Target Variable:

Outcome: This binary variable (0 or 1) represents the diabetes diagnosis.

o	0 indicates that the patient is not diabetic.

o	1 indicates that the patient is diabetic.

Data Source:

This dataset is publicly available and commonly used for machine learning and statistical analysis. It is sourced from the Pima Indians Diabetes Database available on the UCI Machine Learning Repository.

 

5.	Data Preprocessing 

Data preprocessing is a crucial step in preparing the raw data for training machine learning models. It involves cleaning and transforming the data to make it suitable for analysis. Below are the steps followed in the preprocessing of the diabetes dataset:

1. Loading the Data

The data was loaded into the program using the pandas library, which allows easy manipulation of the data. The dataset was stored in a CSV file format.

 

2. Handling Missing Values

We checked for any missing or null values in the dataset. Missing values can affect model accuracy, so we handled them by replacing them with the mean value of each column.

 

In this case, there were no missing values, so no further action was needed.

3. Removing Outliers

Outliers are values that are far outside the normal range. They can negatively impact the performance of the model. We checked for outliers using boxplots and removed any extreme values found.

 



4. Feature Scaling

To make sure all features contribute equally to the model, we scaled numerical values using Min-Max scaling. This scaled the data to a range between 0 and 1.

 



5. Splitting the Data

Next, the data was split into features (X), which are the input variables (like Glucose, BMI, etc.), and the target variable (y), which is the label (Outcome).

 





The data was divided into training and testing sets. 80% of the data was used for training, and 20% was used for testing the model.

 

6. Dealing with Imbalanced Data

If the dataset had an unequal number of diabetes (1) and non-diabetes (0) cases, we would have applied techniques to balance the classes. In our dataset, the classes were balanced, so no further action was needed.

 

6.	Exploratory Data Analysis (EDA)  

Exploratory Data Analysis (EDA) is the process of analyzing data sets to summarize their main characteristics. This helps in understanding the structure of the data and preparing it for the machine learning model.



In our project, we performed EDA using statistical techniques and visualizations:



●	We first checked for missing values and unusual data points (outliers).



●	We analyzed the distribution of each feature such as Glucose, BMI, Insulin, and Age to understand their range and frequency.



●	We used histograms, boxplots, and correlation heatmaps to visually explore the relationships between variables.



●	From the correlation matrix, we observed that features like Glucose, BMI, and Age are strongly related to the presence of diabetes.



●	These insights helped us decide which features are most important for building a reliable prediction model.

 

7.	Feature Engineering 

Feature engineering involved selecting and transforming the right inputs to improve model performance.

Steps Taken:

●	Handling Invalid Zeros: Replaced zero values in columns like Glucose, Blood Pressure, BMI, Insulin, and Skin Thickness with the mean of their respective columns, as zero is not medically valid.



●	Feature Scaling: Applied standardization using StandardScaler to bring all features to a similar scale. This improves the accuracy and speed of many machine learning algorithms.



●	Feature Selection: Retained all original features since each had potential predictive value based on domain knowledge and EDA insights.

Outcome:

The dataset became cleaner, more consistent, and ready for model training. This step helped the model learn more effectively.

8.	Model Building  

Models Selected

Logistic Regression

●	Simple and interpretable model.

●	Useful for understanding how each feature affects the prediction.

Random Forest

●	A powerful ensemble method.

●	Handles nonlinear relationships and interactions well.

●	Robust against outliers and missing values.

●	These two models were chosen to evaluate the trade-off between a simple linear model and a more complex, high-performance ensemble method.

 

Data Splitting

●	Train/Test Split: 80% training, 20% testing

●	Stratified Sampling: Ensured the class distribution (positive/negative diabetes cases) remained balanced in both training and testing datasets.

 

●	Model Training & Evaluation

Model	Accuracy	Precision	Recall	F1-Score

Logistic Regression	84%	78%	72%	75%

Random Forest	91%	88%	85%	86%



●	Precision: Percentage of predicted positives that are actual positives

●	Recall: Percentage of actual positives correctly predicted

●	F1-Score: Harmonic mean of precision and recall

9.	Visualization of Results & Model Insights 

1. Confusion Matrix:

●	What It Shows:

A confusion matrix compares predicted values against actual values to show how well the model is classifying the cases as diabetic or non-diabetic.

●	How to Read It:

o	True Positives (TP): Correctly predicted diabetic cases.

o	True Negatives (TN): Correctly predicted non-diabetic cases.

o	False Positives (FP): Non-diabetic cases incorrectly predicted as diabetic.

o	False Negatives (FN): Diabetic cases incorrectly predicted as non-diabetic.

●	Why It's Useful:

Helps you see where the model is making mistakes, such as predicting non-diabetic people as diabetic.



2. ROC Curve:

●	What It Shows:

The ROC curve shows how well the model can distinguish between diabetic and non-diabetic cases. It compares True Positive Rate (TPR) to False Positive Rate (FPR) at different thresholds.

●	How to Read It:

The AUC (Area Under the Curve) is important:

o	A value close to 1 means the model is good at distinguishing between the classes.

o	A value close to 0.5 suggests the model is no better than random guessing.

●	Why It's Useful:

It helps to find the best threshold for classifying diabetic and non-diabetic cases.





3. Feature Importance:

●	What It Shows:

A feature importance plot shows which features (e.g., glucose, age, BMI) are most important in predicting diabetes.

●	How to Read It:

o	The higher the feature’s importance, the more it contributes to the prediction.

●	Why It's Useful:

Helps you understand what factors are driving the model’s predictions.



4. Residual Plot:

●	What It Shows:

A residual plot shows the difference between the predicted values and the actual values. It helps to check if there are any patterns the model is missing.

●	How to Read It:

o	If the residuals are randomly scattered around 0, the model is good.

o	If there are patterns, it means the model is missing something.

●	Why It's Useful:

Helps you see if the model is making systematic errors that need to be fixed.



5. Model Performance Comparison:

●	What It Shows:

A bar chart comparing the performance of different models (like Logistic Regression and Random Forest) based on metrics like accuracy, precision, recall, and F1-score.

●	How to Read It:

o	The model with the highest values for accuracy, precision, and recall is typically the best.

●	Why It's Useful:

Helps you choose the best model based on performance.



10.	Tools and Technologies Used 

  Programming Language:

●	Python

  Development Environments:

●	VS Code

  Libraries and Frameworks:

●	pandas (for data manipulation)

●	numpy (for numerical computations)

●	matplotlib (for basic data visualization)

●	seaborn (for advanced data visualization)

●	scikit-learn (for machine learning algorithms)

●	Flask (for backend API development)

●	pickle (for saving and loading the model)

●	firebase-admin (for database interaction)

  Database:

●	Firebase (for storing user input and prediction results)

 

 

11.	Team Members and Contributions 

KALPANA V:

Model Development and Evaluation: Built and evaluated various machine learning models, including Logistic Regression and Random Forest, and performed model evaluation using metrics such as accuracy, precision, recall, and F1-score.

JASMINE R:

Exploratory Data Analysis (EDA): Performed data analysis to identify patterns, trends, and outliers in the dataset. Created visualizations using libraries such as Seaborn and Matplotlib.

JEYAUPASHANA K.J:

Data Cleaning: Responsible for cleaning and preprocessing the dataset, handling missing values, and performing necessary transformations to prepare the data for model building.

JANANI M:

Feature Engineering: Created new features, transformed existing ones, and performed feature scaling to enhance the performance of the model.



 
