import pandas as pd
import pickle
import json
import os
import time
import subprocess
import subprocess
import pkg_resources
##################Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])


##################Function to get model predictions
# def model_predictions():


def model_predictions(data):
    # Load the configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load the deployed model
    model_path = os.path.join(config['prod_deployment_path'], 'trainedmodel.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Make predictions using the deployed model
    predictions = model.predict(data)

    return predictions.tolist()  # Return as list


##################Function to get summary statistics
def dataframe_summary():
    # Load the configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load the dataset
    data_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
    data = pd.read_csv(data_path)

    # Calculate summary statistics
    means = data.mean(numeric_only=True).tolist()
    medians = data.median(numeric_only=True).tolist()
    std_devs = data.std(numeric_only=True).tolist()

    # Combine the results into a list of lists
    summary_stats = {
        'means': means,
        'medians': medians,
        'std_devs': std_devs
    }

    return summary_stats


def missing_data():
    # Load the configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    # Load the dataset
    data_path = os.path.join(config['output_folder_path'], 'finaldata.csv')
    data = pd.read_csv(data_path)

    # Calculate the percentage of missing values for each column
    missing_percentages = (data.isna().sum() / len(data)) * 100

    return missing_percentages.tolist()  # Return as list


##################Function to get timings


def execution_time():
    # Time data ingestion
    start_time = time.time()
    subprocess.run(['python3', 'ingestion.py'])
    ingestion_time = time.time() - start_time

    # Time model training
    start_time = time.time()
    subprocess.run(['python3', 'training.py'])
    training_time = time.time() - start_time

    return [ingestion_time, training_time]

##################Function to check dependencies
def outdated_packages_list():
    # Load the current dependencies from requirements.txt
    with open('requirements.txt', 'r') as file:
        requirements = file.readlines()

    dependencies = [line.strip().split('==')[0] for line in requirements]

    # Initialize the result list for dependencies and versions
    dependency_info = []

    # Get a list of outdated packages using pip list --outdated
    result = subprocess.run(
        ['pip', 'list', '--outdated', '--format=freeze'],
        stdout=subprocess.PIPE,
        text=True
    )

    # Parse the output of pip list --outdated
    outdated_packages = result.stdout.splitlines()

    # Store outdated package info in a dictionary
    outdated_dict = {}
    for package_info in outdated_packages:
        package, current_version, new_version = package_info.split('==')[0], package_info.split('==')[1], \
        package_info.split('->')[-1].strip()
        outdated_dict[package] = new_version

    # Check each dependency's current and latest version
    for dep in dependencies:
        try:
            # Get current installed version
            current_version = pkg_resources.get_distribution(dep).version

            # Check if the package is outdated
            latest_version = outdated_dict.get(dep, current_version)  # If not outdated, latest = current version

            dependency_info.append([dep, current_version, latest_version])

        except pkg_resources.DistributionNotFound:
            dependency_info.append([dep, "Not installed", "Unknown"])

    return dependency_info


# get a list of
if __name__ == '__main__':
    file=os.path.join(test_data_path, 'testdata.csv')
    test_data = pd.read_csv(file)
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_data['exited']
    model_predictions(X_test)
    summary_stats = dataframe_summary()
    print(summary_stats)
    mdata = missing_data()
    print(mdata)
    start,finish =execution_time()
    print(start,finish)
    dependencies = outdated_packages_list()
    for dep in dependencies:
     print(f"Package: {dep[0]}, Current version: {dep[1]}, Latest version: {dep[2]}")
