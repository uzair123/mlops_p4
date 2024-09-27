#from flask import Flask
#session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

#dataset_csv_path = os.path.join(config['output_folder_path'])
#model_path = os.path.join(config['output_model_path'])

output_folder_path = config['output_folder_path']
output_model_path = config['output_model_path']
#################Function for training the model
def train_model():
    # Step 1: Load dataset
    data_path = os.path.join(output_folder_path, 'finaldata.csv')
    data = pd.read_csv(data_path)

    # Step 2: Prepare features and target variable
    X = data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]  # Features
    y = data['exited']  # Target variable

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print("Model training complete. Trained model saved to", output_model_path)

    #use this logistic regression for training
    #model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
    #                intercept_scaling=1, l1_ratio=None, max_iter=100,
    #                multi_class='multinomial', n_jobs=None, penalty='l2',
    #                random_state=0, solver='liblinear', tol=0.0001, verbose=0,
    #                warm_start=False)
    model=LogisticRegression()
    #fit the logistic regression to your data
    # Step 4: Train a Logistic Regression model
    #model = LogisticRegression()
    model.fit(X_train, y_train)
    #write the trained model to your workspace in a file called trainedmodel.pkl
    # Step 5: Save the trained model to a file
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)

if __name__ == '__main__':
    train_model()