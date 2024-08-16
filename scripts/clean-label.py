import os

import pandas as pd

csv_path = '../../volatile/cgr_meta_data_adjusted.csv'

df = pd.read_csv(csv_path)

data_folder = '../../volatile/cgr/'

# discover all the files in the folder
files = os.listdir(data_folder)

# add the column called 'file_name' to the dataframe where the filename is found by matching the genome_id as a substring of the filenames
df['file_name'] = df['genome_id'].apply(lambda x: [file for file in files if '_' + x + '.' in file or '_' + x + '_' in file])
# check for duplicates
dup = df[df['file_name'].apply(lambda x: len(x) > 1)][['genome_id', 'file_name']]
df['file_name'] = df['file_name'].apply(lambda x: x[0] if len(x) > 0 else None)

# drop the rows where the file_name is None
df = df.dropna(subset=['file_name'])

df.to_csv('../../volatile/cgr_label.csv', index=False)

print(df)