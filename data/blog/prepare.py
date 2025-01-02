import os
import requests
import tiktoken
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(__file__), exist_ok=True)

# Download the Blog Authorship Corpus if not already present
data_path = os.path.join(os.path.dirname(__file__), 'blogtext.csv')
if not os.path.exists(data_path):
    print("Downloading Blog Authorship Corpus...")
    url = "https://raw.githubusercontent.com/rbroc/text-mining-course/master/data/blogtext.csv"
    response = requests.get(url, stream=True)
    with open(data_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)

# Load and prepare the data
df = pd.read_csv(data_path, encoding='utf-8')

# Print column names to debug
print("Available columns:", df.columns.tolist())

# Use 'post' instead of 'text'
text = "\n".join(df['post'].astype(str))

# Split into train and validation sets (90/10 split)
train_data, val_data = train_test_split(
    text.split('\n'), 
    test_size=0.1, 
    random_state=42
)

# Rejoin the splits
train_data = '\n'.join(train_data)
val_data = '\n'.join(val_data)

# Encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# Export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin')) 