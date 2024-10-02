import requests
import json
import os


def call_api():
    # Load config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    api_base = 'http://127.0.0.1:5001'

    # Call prediction endpoint
    test_data_path = os.path.join(config['test_data_path'], 'testdata.csv')
    prediction_response = requests.post(f'{api_base}/prediction', json={"data_path": test_data_path})
    predictions = prediction_response.json()

    # Call scoring endpoint
    scoring_response = requests.get(f'{api_base}/scoring')
    f1_score = scoring_response.json()['f1_score']

    # Call summary statistics endpoint
    stats_response = requests.get(f'{api_base}/summarystats')
    summary_stats = stats_response.json()

    # Call diagnostics endpoint
    diagnostics_response = requests.get(f'{api_base}/diagnostics')
    diagnostics_results = diagnostics_response.json()

    # Combine the outputs
    combined_results = {
        "predictions": predictions,
        "f1_score": f1_score,
        "summary_stats": summary_stats,
        "diagnostics": diagnostics_results
    }

    # Save the results to apireturns.txt
    output_file_path = os.path.join(config['output_model_path'], 'apireturns.txt')
    with open(output_file_path, 'w') as output_file:
        output_file.write(json.dumps(combined_results, indent=4))


if __name__ == "__main__":
    call_api()




