import pandas as pd
import re
from ftfy import fix_text

def normalize_text(text: str) -> str:
    text = fix_text(text, normalization='NFKC')
    text = re.compile(r'https?://\S+|www\.\S+').sub('[URL]', text)
    text = re.compile(r'\S+@\S+').sub('[EMAIL]', text)

    return text.strip()

def split_text_to_chunks(df, text_col='instruction', label_col='label', category_col='category', max_chars=1200):
    new_rows = []
    
    for _, row in df.iterrows():
        text = str(row[text_col])
        label = row[label_col]
        category = row[category_col]
        
        if len(text) > max_chars:
            chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars - 200)]
            for chunk in chunks:
                new_rows.append({text_col: chunk, category_col: category, label_col: label, 'is_chunked': True})
        else:
            new_rows.append({text_col: text, category_col: category, label_col: label, 'is_chunked': False})
            
    return pd.DataFrame(new_rows)


import math

