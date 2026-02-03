import pandas as pd
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from config import settings

os.makedirs('data', exist_ok=True)

ds = load_dataset(settings.DATASET_PATH)

df = pd.DataFrame(ds['train'])
categories = ['classification', 'summarization', 'creative_writing', 'general_qa']
label_map = {
    'classification': 0,
    'summarization': 0,
    'creative_writing': 1,
    'general_qa': 1
}
df_filtered = df[df['category'].isin(categories)].copy()
df_filtered['label'] = df_filtered['category'].map(label_map)
data = df_filtered[['instruction', 'label', 'category']]
print('Len of dataframe: ', len(data))
print(data['label'].value_counts())

max_len = df['instruction'].str.len().max()
print(f"max length of instruction: {max_len}")

train_df, test_df = train_test_split(
    data, 
    test_size=0.2, 
    random_state=settings.SEED, 
    stratify=data['label']
)

train_df.to_csv('data/data_train.csv', index=False)
test_df.to_csv('data/data_test.csv', index=False)