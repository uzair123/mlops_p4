#pytfrom flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

output_model_path = config['output_model_path']
prod_deployment_path = config['prod_deployment_path']
ingested_data_path = config['output_folder_path']

####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    # Step 1: Define files to be copied
    files_to_copy = ['trainedmodel.pkl', 'latestscore.txt']

    # Step 2: Copy files to production deployment path
    for file in files_to_copy:
        src = os.path.join(output_model_path, file)
        dst = os.path.join(prod_deployment_path, file)
        shutil.copy(src, dst)

    src = os.path.join(ingested_data_path, 'ingestedfiles.txt')
    dst = os.path.join(prod_deployment_path, 'ingestedfiles.txt')
    shutil.copy(src, dst)
    print(f"Deployment complete. Files copied to {prod_deployment_path}.")

        
        
if __name__ == '__main__':
    store_model_into_pickle()
