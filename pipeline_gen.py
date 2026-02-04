import argparse
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from config import settings
from datetime import datetime
from utils import *
from dotenv import load_dotenv
load_dotenv()

def truncateDim(X):
    X = X[:, :settings.LATENT_DIM]
    X = F.normalize(X, p=2, dim=1)
    return X

def predict_prompt(text):
    emb = model.encode([text])
    label = pipe.predict(emb)[0]
    prob = pipe.predict_proba(emb)[0]
    return label, prob

def get_voting_prediction(text, max_chars=1200, overlap=200):
    if len(text) <= max_chars:
        chunks = [text]
    else:
        chunks = []
        step = max_chars - overlap
        for i in range(0, len(text), step):
            chunks.append(text[i : i + max_chars])
            if i + max_chars >= len(text): break
    chunk_embs = model.encode(chunks, show_progress_bar=False)
    if model_setting == 'nomic-embed-text-v1.5':
        chunk_embs = truncateDim(torch.tensor(chunk_embs)).numpy()
    
    all_chunk_probs = pipe.predict_proba(chunk_embs) 
    
    avg_probs = all_chunk_probs.mean(axis=0)
    avg_label_1_prob = avg_probs[1]

    if a.use_threshold:
        final_label = 1 if avg_label_1_prob >= settings.THRESHOLD else 0
    else:
        final_label = np.argmax(avg_probs)
    
    final_confidence = avg_probs[final_label]
    
    return final_label, final_confidence

parser = argparse.ArgumentParser()
parser.add_argument('--use_pca', action='store_true', help='use latent or not')
parser.add_argument('--use_threshold', action='store_true', help='use confidence threshold or not')
a = parser.parse_args()

pca_setting = str(settings.LATENT_DIM) if a.use_pca else 'None'
t_setting = str(settings.THRESHOLD) if a.use_threshold else 'None'
model_setting = os.path.basename(settings.ENCODER_NAME)
seed_setting = str(settings.SEED)

os.makedirs('reports/router_test', exist_ok=True)

# read training set
train_df = pd.read_csv('data/data_train.csv')
train_df = split_text_to_chunks(train_df)
train_df['instruction'] = train_df['instruction'].apply(normalize_text)
train_df = train_df[train_df['instruction'].str.strip().astype(bool)]
model = SentenceTransformer(settings.ENCODER_NAME, trust_remote_code=True)

encoded_path = Path(f'data/data_train_{model_setting}_norm_encoded.npy')
if not encoded_path.exists():
    print('######### Encoding Training Set... #########')
    X_train = model.encode(train_df['instruction'].tolist(), show_progress_bar=True)
    np.save(encoded_path, X_train)
else:
    print('######### Loading Encoded Training Set... #########')
    X_train = np.load(encoded_path)

if model_setting == 'nomic-embed-text-v1.5':
    X_train = truncateDim(X_train)

y_train = train_df['label'].values
print('X_train.shape: ', X_train.shape)
original_dim = X_train.shape[1]

# two pipelines (use_pca or not)
if a.use_pca:
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=settings.LATENT_DIM, random_state=settings.SEED)),
        ('classifier', LogisticRegression(class_weight='balanced', random_state=settings.SEED))
    ])
else:
    pipe = Pipeline([
        ('classifier', LogisticRegression(class_weight='balanced', random_state=settings.SEED))
    ])

# training model ...
pipe.fit(X_train, y_train)

if a.use_pca:
    explained_var = np.sum(pipe.named_steps['pca'].explained_variance_ratio_)
    print(f"LATENT_DIM = {settings.LATENT_DIM} preserves {explained_var:.2%} of original data.")  

# read testing set
test_df = pd.read_csv('data/data_test.csv')
test_df['instruction'] = test_df['instruction'].apply(normalize_text)

print('######### Encoding Test Set... #########')

y_test_voting = []
y_pred_voting = []
categories_voting = []
instructions_voting = []
confidence_voting = []

# evaluating ...
for _, row in test_df.iterrows():
    raw_text = str(row['instruction'])
    actual_label = row['label']
    category = row['category']
    
    predicted_label, confidence = get_voting_prediction(raw_text)
    
    y_test_voting.append(actual_label)
    y_pred_voting.append(predicted_label)
    confidence_voting.append(confidence)
    categories_voting.append(category)
    instructions_voting.append(raw_text)

results_df = pd.DataFrame({
    'instruction': instructions_voting,
    'category': categories_voting,
    'confidence': confidence_voting,
    'label_true': y_test_voting,
    'label_predict': y_pred_voting
})

# Accuracy of each category
for cat in results_df['category'].unique():
    subset = results_df[results_df['category'] == cat]
    accuracy = (subset['label_true'] == subset['label_predict']).mean()
    print(f"Accuracy of Category [{cat:25}]: {accuracy:.2%}")

# misclassification summarization samples
summarization_errors = results_df[
    (results_df['category'] == 'summarization') & 
    (results_df['label_true'] != results_df['label_predict'])
]
print("The first 5 misclassify summarization data: ")
print(test_df.loc[summarization_errors.index, 'instruction'].head(5))

# results
print("\n######### Evaluation #########")
print(classification_report(y_test_voting, y_pred_voting))

test_text = "Can you summarize the main points of the French Revolution?"
label, prob = predict_prompt(test_text)
print(f"\nTest Input: {test_text}")
print(f"Classification Result: {'Slow Path (1)' if label == 1 else 'Fast Path (0)'}")
print(f"Confidence: [Label 0: {prob[0]:.4f}, Label 1: {prob[1]:.4f}]")

filename = f"reports/router_test/eval_{model_setting}_pca{pca_setting}_t{t_setting}_sd{seed_setting}.log"

# output evaluation results
with open(filename, 'w', encoding='utf-8') as f:
    f.write(f"Test Time: {datetime.now()}\n")
    f.write(f"Embedding Model: {settings.ENCODER_NAME}\n")
    f.write(f"Original Dimensions: {original_dim}\n")
    f.write(f"PCA Dimensions: {settings.LATENT_DIM if a.use_pca else 'None'}\n")
    f.write(f"Threshold: {settings.THRESHOLD if a.use_threshold else 'None'}\n")
    #f.write(f"Accuracy of test set: {score:.4f}\n")
    if a.use_pca:
        f.write("-" * 30 + "\n")
        f.write(f"######### PCA #########\n")
        f.write(f"LATENT_DIM = {settings.LATENT_DIM} preserves {explained_var:.2%} of original data.\n")
    f.write("-" * 30 + "\n")
    for cat in results_df['category'].unique():
        subset = results_df[results_df['category'] == cat]
        accuracy = (subset['label_true'] == subset['label_predict']).mean()
        f.write(f"Accuracy of Category [{cat:25}]: {accuracy:.2%}]\n")

    f.write("The first 5 misclassify summarization data: \n")
    f.write(test_df.loc[summarization_errors.index, 'instruction'].head(5).to_string() + "\n")
    f.write("-" * 30 + "\n")
    f.write("######### Evaluation #########\n")
    f.write(classification_report(y_test_voting, y_pred_voting))
    f.write("-" * 30 + "\n")
    f.write(f"Test Input: {test_text}\n")
    f.write(f"Classification Result: {'Slow Path (1)' if label == 1 else 'Fast Path (0)'}\n")
    f.write(f"Confidence: [Label 0: {prob[0]:.4f}, Label 1: {prob[1]:.4f}]\n")

print('######### Saving Semantic Router... #########')
joblib.dump(pipe, f'model/semantic_router_{model_setting}.joblib')
print('######### Completed. #########')