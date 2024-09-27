import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    # Step 1: Automatically detect all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder_path) if f.endswith('.csv')]

    # Step 2: Read all CSV files and combine into a single DataFrame
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(input_folder_path, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)

    # Combine all dataframes into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Step 3: Remove duplicates
    final_df = combined_df.drop_duplicates()

    # Step 4: Save the deduplicated DataFrame to 'finaldata.csv' in the output folder
    output_file_path = os.path.join(output_folder_path, 'finaldata.csv')
    final_df.to_csv(output_file_path, index=False)

    # Step 5: Save the list of ingested files to 'ingestedfiles.txt' in the output folder
    ingested_files_path = os.path.join(output_folder_path, 'ingestedfiles.txt')
    with open(ingested_files_path, 'w') as f:
        for file in csv_files:
            f.write(file + '\n')

    print(f"Data ingestion complete. {len(csv_files)} files processed.")


if __name__ == '__main__':
    merge_multiple_dataframe()
