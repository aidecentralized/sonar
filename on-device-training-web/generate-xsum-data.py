import os
import pandas as pd
import json
from datasets import load_dataset
from sklearn.model_selection import train_test_split

data_dir = './web/data/.xsum'
train_file = 'xsum-train.json'
test_file = 'xsum-test.json'
val_file = 'xsum-validation.json'

os.makedirs(data_dir, exist_ok=True) #create xsum dir if not exist

dataset = load_dataset("xsum")

df = pd.DataFrame(dataset['train'])

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

def save_to_json(dataframe, file_path):
    data = []
    for _, row in dataframe.iterrows():
        item = {
            'text': row['document'],
            'summary': row['summary']
        }
        data.append(item)
    with open(file_path, 'w') as f:
        json.dump(data, f)

save_to_json(train_df, os.path.join(data_dir, train_file))
save_to_json(test_df, os.path.join(data_dir, test_file))
save_to_json(val_df, os.path.join(data_dir, val_file))
