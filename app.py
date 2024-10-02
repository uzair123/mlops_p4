from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
import diagnostics
#import predict_exited_from_saved_model
import json
import os
import scoring

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

#prediction_model = None

# Prediction endpoint
@app.route("/prediction", methods=['POST'])
def predict():
    # Get file path from request
    data_path = request.json.get('data_path')

    # Read test data
    test_data = pd.read_csv(data_path)

    # Get predictions
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    predictions = diagnostics.model_predictions(X_test)

    return jsonify(predictions), 200


# Scoring endpoint
@app.route("/scoring", methods=['GET'])
def score():
    f1_score = scoring.score_model()

    return jsonify({"f1_score": f1_score}), 200


# Summary statistics endpoint
@app.route("/summarystats", methods=['GET'])
def summarystats():
    summary_stats = diagnostics.summary_statistics()

    return jsonify(summary_stats), 200


# Diagnostics endpoint
@app.route("/diagnostics", methods=['GET'])
def diagnostics_endpoint():
    # Perform timing, missing data, and dependency checks
    timing_info = diagnostics.execution_time()
    missing_data_info = diagnostics.missing_data()
    dependency_info = diagnostics.check_dependencies()

    diagnostics_results = {
        "timing": timing_info,
        "missing_data": missing_data_info,
        "dependencies": dependency_info
    }

    return jsonify(diagnostics_results), 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
