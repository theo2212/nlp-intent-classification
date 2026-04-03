import os
import pandas as pd
from datasets import load_dataset

# Local cache paths
CACHE_DIR = "cache"
LLM_CACHE_FILE = os.path.join(CACHE_DIR, "llm_predictions_cache.parquet")

def load_banking_dataset():
    """
    Loads MTEB/Banking77 dataset from Hugging Face.
    Returns the train and test splits as Pandas DataFrames.
    """
    print("Loading MTEB/Banking77 dataset...")
    dataset = load_dataset("mteb/banking77")
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    
    label_names = sorted(df_train['label_text'].unique().tolist())
    
    return df_train, df_test, label_names

def init_cache():
    """Initializes cache directory and empty parquet files if they don't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        
    if not os.path.exists(LLM_CACHE_FILE):
        # Create an empty DataFrame for caching Gemini responses
        # Columns: text_hash (or raw text), prediction
        empty_llm = pd.DataFrame(columns=['text', 'predicted_label'])
        empty_llm.to_parquet(LLM_CACHE_FILE, engine='pyarrow')

def load_llm_cache():
    """Loads the LLM prediction cache from disk."""
    if not os.path.exists(LLM_CACHE_FILE):
        init_cache()
    return pd.read_parquet(LLM_CACHE_FILE, engine='pyarrow')

def save_llm_cache(df):
    """Saves the LLM prediction cache to disk."""
    df.to_parquet(LLM_CACHE_FILE, engine='pyarrow')

def get_cached_prediction(text, cache_df):
    """
    Looks up a question's predicted label in the cache.
    Returns the label if found, otherwise None.
    """
    match = cache_df[cache_df['text'] == text]
    if not match.empty:
        return match.iloc[0]['predicted_label']
    return None

def add_prediction_to_cache(text, predicted_label, cache_df):
    """
    Adds a new prediction to the local DataFrame. Note: User must call `save_llm_cache` manually to persist.
    """
    new_row = pd.DataFrame([{'text': text, 'predicted_label': predicted_label}])
    if not new_row.isna().all().all():
        # Avoid concatenation error if empty or bad row
        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
    return cache_df
