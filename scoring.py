#from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
from sklearn.metrics import f1_score


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

#dataset_csv_path = os.path.join(config['output_folder_path'])
#test_data_path = os.path.join(config['test_data_path'])

test_data_path = config['test_data_path']
output_model_path = config['output_model_path']

#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    # Step 1: Load test dataset
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_data['exited']

    # Step 2: Load trained model
    model_path = os.path.join(output_model_path, 'trainedmodel.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Step 3: Make predictions and calculate F1 score
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    # Step 4: Save F1 score to latestscore.txt
    score_path = os.path.join(output_model_path, 'latestscore.txt')
    with open(score_path, 'w') as score_file:
        score_file.write(str(f1))

    print(f"F1 score: {f1}. Score saved to {score_path}.")

if __name__ == '__main__':
    score_model()
