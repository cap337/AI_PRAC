import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def extract_features_and_labels(csv_file_info, csv_file_stats):
    # Load the dataset
    player_info = pd.read_csv(csv_file_info)
    player_stats = pd.read_csv(csv_file_stats)

    player_info_cleaned = player_info[['display_first_last', 'birthdate', 'height', 'weight', 'position', 'draft_number']]
    player_info_cleaned = player_info_cleaned.rename(columns={'display_first_last':'player'})
    player_stats_cleaned = player_stats.drop(columns = ['seas_id','player_id', 'pos', 'lg'])

    data = pd.merge(player_stats_cleaned,player_info_cleaned,on='player',how='inner')
    data = data.dropna()
    data = data.sort_values(by=['player', 'season'])

    grouped = data.groupby('player')
    input_rows = []
    output_rows = []

    for player, group in grouped: 
        if len(group) > 1:
            for i in range(len(group) - 1):
                input_rows.append(group.iloc[i])
                output_rows.append(group.iloc[i + 1])

    input_rows_df = pd.DataFrame(input_rows)
    output_rows_df = pd.DataFrame(output_rows)

    # additional feature selection (dropping more columns)
    input_rows_df = input_rows_df.drop(columns=['player','birth_year','tm', 'draft_number', 'birthdate'])
    output_rows_df = output_rows_df.drop(columns=['player','birth_year','tm', 'draft_number', 'birthdate'])

    # converting hieght column to inches 
    def height_to_inches(height_str):
        if isinstance(height_str, str):
            feet, inches = map(int, height_str.split('-'))
            return feet * 12 + inches 
        else: 
            return float('nan')

    input_rows_df['height'] = input_rows_df['height'].apply(height_to_inches)
    input_rows_df['height'] = input_rows_df['height'].astype(float)
    output_rows_df['height'] = output_rows_df['height'].apply(height_to_inches)
    output_rows_df['height'] = output_rows_df['height'].astype(float)

    label_encoder = LabelEncoder()

    for col in input_rows_df.columns:
        if input_rows_df[col].dtype == 'object':
            input_rows_df[col] = label_encoder.fit_transform(input_rows_df[col])

    int_columns = input_rows_df.select_dtypes(include=['int']).columns
    input_rows_df[int_columns] = input_rows_df[int_columns].astype(float)

    for col in output_rows_df.columns:
        if output_rows_df[col].dtype == 'object':
            output_rows_df[col] = label_encoder.fit_transform(output_rows_df[col])

    int_columns = output_rows_df.select_dtypes(include=['int']).columns
    output_rows_df[int_columns] = output_rows_df[int_columns].astype(float)

    input_rows_array = input_rows_df.to_numpy()
    output_rows_array = output_rows_df.to_numpy()


    return input_rows_array, output_rows_array