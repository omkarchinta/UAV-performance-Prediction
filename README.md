Project task:
Project Title: Machine Learning for UCAV Performance Prediction
Project Overview: The objective of this project is to leverage machine learning (ML) to predict the performance of Unmanned Combat Aerial Vehicles (UCAVs) under varying flight conditions. The project will use an existing UCAV flight data database to train models that predict the aircraft's behavior, such as stability margins and control system adjustments, based on dynamic parameters like center of gravity (CG), wind disturbances, and flight maneuvers.
Project Phases:
1. Data Collection from Existing Database
Objective: Use an existing UCAV flight data database to collect relevant flight data for training ML models. The database should contain simulations or real-world flight data for various UCAV missions, environmental conditions, and flight maneuvers.

Steps:

Access the Database: Obtain access to an existing UCAV flight data repository (e.g., flight simulators, historical flight logs).
Identify Key Data Features:
Flight Parameters: Altitude, speed, pitch, yaw, roll, etc.
Control Inputs: Thrust, control surface deflections, etc.
Environmental Factors: Wind speed/direction, turbulence, temperature, etc.
Aircraft Conditions: Center of gravity (CG), fuel load, weight distribution, etc.
Data Extraction & Preprocessing:
Filter and extract relevant data based on project objectives.
Perform preprocessing steps like data cleaning, normalization, and handling missing values.
Deliverables:

Processed dataset ready for training and testing ML models.
Data preprocessing scripts for cleaning and preparing the dataset.
2. Supervised Learning for UCAV Performance Prediction
Objective: Develop supervised learning models to predict UCAV performance based on the input parameters collected in the previous phase. This includes predicting stability margins, control system adjustments, and overall flight performance under different conditions.

Steps:

Data Preparation:
Split the data into training, validation, and test sets.
Perform feature engineering (e.g., creating time-series features, interaction terms, or aggregating flight parameters).
Model Selection & Training:
Regression Models: Train regression models (e.g., Linear Regression, Random Forest Regression, or Support Vector Regression) to predict continuous values such as stability margin, control surface deflections, or control adjustments required.
Classification Models: Train classification models (e.g., Decision Trees, Random Forests, or Neural Networks) if the goal is to classify flight conditions as stable or unstable, or to predict discrete control adjustments.
Model Evaluation:
Evaluate model performance using appropriate metrics:
Regression Models: Mean Squared Error (MSE), R², etc.
Classification Models: Accuracy, Precision, Recall, F1-Score, etc.
Perform cross-validation to ensure robust model generalization.
Deliverables:

Trained ML models for predicting UCAV performance (e.g., stability, control adjustments).
Evaluation report with performance metrics for each model.
________________________________________
Solution


The project aimed to leverage machine learning (ML) techniques to predict the performance of Unmanned Combat Aerial Vehicles (UCAVs) using flight data from a scaled model of the Cub Crafters CC11-100 Sport Cub S2 aircraft. The main objectives were to predict stability margins and control system adjustments under various conditions by training ML models on a comprehensive flight dataset.
________________________________________
1. Data Collection and Preprocessing
Objective:
The primary goal was to extract and prepare the relevant data from the UCAV flight data database for machine learning (ML) modeling.
Data Source and Characteristics:
•	Aircraft Model: The dataset used data from a 26% scale model of the Cub Crafters CC11-100 Sport Cub S2, which has a high-wing configuration and an electric propulsion system.
•	Instrumentation and Data Acquisition: Data was recorded using an Al Volo FDAQ data acquisition system operating at 400 Hz, capturing parameters such as linear/angular accelerations, airspeed, control surface deflections, motor RPM, etc.
Data set link:
http://uavdb.org/cub.php

Data Loading:
The dataset was initially provided in multiple .mat files. I combined all of them and saved the data as a pandas DataFrame for easier processing.
Data Preprocessing Steps:
•	Data Type Conversion: Initially, all column values were of complex number type. I converted them to real numbers for further analysis.
•	Cleaning: Outliers and missing data were handled through appropriate imputation and exclusion techniques. Missing values in highly correlated columns were filled using linear interpolation. In cases where both correlated columns had missing values, I used the iterative imputer to fill them. For the remaining columns, missing values were filled based on their distribution.
•	Outlier Treatment: Some outliers were detected during data exploration. However, I did not remove them, as they could represent real-world scenarios.
•	Handling Missing Data: Columns with more than 70% missing values were not dropped, assuming that they might provide useful information during analysis. However, I did drop highly correlated columns.
•	Final Dataset: After processing, I ensured that the data was ready for ML model training and testing.
Deliverables:
•	A processed dataset ready for ML model training and testing.
•	Data preprocessing scripts for cleaning and data preparation.
________________________________________
2. Supervised Learning for UCAV Performance Prediction
Objective:
The objective was to develop machine learning models to predict UAV performance indicators, including stability margins and control surface adjustments under various flight conditions.
Model Development and Training:
•	The cleaned and processed data was split into training and testing sets, with feature engineering applied to optimize prediction accuracy.
Model Selection:
•	Regression Models:
o	Linear Regression: Applied to predict continuous metrics such as stability margins and control surface deflections.
o	MLP (Multi Layer Perceptron): Used for moderate accuracy prediction but at higher computational cost.
o	LightGBM: A gradient boosting model that captured complex relationships between input parameters and target outputs effectively.
•	Classification Models:
o	Random Forest: Trained to classify flight conditions, such as stable vs. unstable states.
o	Logistic Regression: A simple classifier tested for complex classifications but required more tuning for optimal performance.
o	XGBoost: A powerful model used to classify flight conditions with great accuracy.



Model Evaluation Results:
•	Regression Performance:

o	Linear Regression: Demonstrated low accuracy and showed limitations due to the non-linear interactions in the dataset.
o	LightGBM: Achieved better accuracy in predicting continuous variables, capturing complex relationships effectively.


	Flap Deflection (Flap_defl):
	R² Score: 0.8937
	Mean Squared Error (MSE): 7.8498
	Rudder Deflection (Rudder_defl):
	R² Score: 0.7815
	Mean Squared Error (MSE): 2.2258
	Aileron Deflection (Aileron_defl):
	R² Score: 0.7519
	Mean Squared Error (MSE): 0.4333


o	MLP: Moderate accuracy but with high computational cost.
•	Classification Performance:


o	XGBoost: Accurately classified flight conditions with high F1-score, precision, and accuracy, reflecting strong generalization capabilities.
	Best Hyperparameters:
	subsample: 0.875
	n_estimators: 100
	max_depth: 8
	learning_rate: 0.0464
	gamma: 0.001
	colsample_bytree: 0.875
	Accuracy: 0.9953



	Classification Report:
Best Parameters: {'subsample': 0.875, 'n_estimators': 100, 'max_depth': 8, 'learning_rate': 0.046415888336127774, 'gamma': 0.001, 'colsample_bytree': 0.875}
Accuracy: 0.9953225066118774

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     28711
           1       0.99      1.00      0.99     17040

    accuracy                           1.00     45751
   macro avg       0.99      1.00      1.00     45751
weighted avg       1.00      1.00      1.00     45751


o	Logistic Regression: Showed potential for complex classifications but required further hyperparameter tuning for optimal results.
Cross-Validation:
I performed cross-validation to ensure robust model generalization. Consistent performance was observed across different folds of the dataset.
Key Metrics:
•	Regression Models: The Mean Squared Error (MSE) and R² scores varied across models, with LightGBM offering the best balance of accuracy and interpretability.
•	Classification Models: The classification models, especially XGBoost, exhibited high accuracy and F1-scores, demonstrating a strong ability to distinguish between stable and unstable flight conditions.
________________________________________
3. Conclusions and Recommendations
The project successfully developed and evaluated machine learning models for predicting UAV performance metrics. Among the regression models, LightGBM demonstrated the best performance in predicting stability margins and control surface adjustments. For classification tasks, XGBoost showed remarkable accuracy in identifying stable vs. unstable flight conditions.
However, there is room for further improvement. Additional work, such as hyperparameter optimization and potentially the exploration of more complex architectures (e.g., deep learning models), could enhance prediction accuracy and robustness.
The approach presented here demonstrates strong potential for improving UAV control and stability analysis, providing valuable predictive insights that are critical for real-world UAV applications.
Note: I faced challenges in predicting higher accuracy for total control deflections, particularly for the Elevator Deflection (Elevator_defl), where the R² score was less than 0.5. I was able to achieve reasonable accuracy for individual control deflections such as Aileron_defl, Rudder_defl, and Flap_defl. Despite trying models like KNN and SVM, they proved computationally challenging. More analysis of UAV stability, along with other performance metrics, could further improve the model's overall effectiveness, but time constraints(one day) limited these additional analyses.

