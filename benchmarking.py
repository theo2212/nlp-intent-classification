import time
import random
import os
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from dotenv import load_dotenv
from data_loader import load_banking_dataset, load_llm_cache
from classical_ml import ClassicalMLPipeline
from llm_pipeline import LLMPipeline

# Load environment variables (API Key)
if not load_dotenv():
    print("Warning: .env file not found. Ensure environment variables are set manually.")

if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
    print("CRITICAL ERROR: No API Key found. Please create a .env file with GOOGLE_API_KEY=your_key")
    # We don't exit here to allow Classical ML to still be tested if needed, 
    # but the LLM part will fail gracefully.

class UnifiedClassifier:
    def __init__(self, classical_pipeline, llm_pipeline, llm_cache):
        self.classical_pipeline = classical_pipeline
        self.llm_pipeline = llm_pipeline
        self.llm_cache = llm_cache

    def predict_intent(self, ticket_text, model_type="classical"):
        """Unified interface for intent prediction."""
        if model_type == "classical":
            start = time.time()
            pred = self.classical_pipeline.predict([ticket_text])[0]
            latency = time.time() - start
            return pred, latency
        elif model_type == "llm":
            # For LLMs, exclude network retry wait times in the latency computation by taking local time 
            # ideally around the API call, but we measure the function's base latency without retries here.
            # If it's cached, latency is near zero. If not, it includes the actual API inference time.
            start = time.time()
            pred, self.llm_cache, _ = self.llm_pipeline.process_ticket(ticket_text, self.llm_cache)
            latency = time.time() - start
            return pred, latency
        else:
            raise ValueError("model_type must be 'classical' or 'llm'")

def run_benchmarking(subsample_size=150):
    print("=== Phase 0: Data Ingestion ===")
    df_train, df_test, label_names = load_banking_dataset()
    cache_df = load_llm_cache()

    print(f"\\n=== Phase 1: Classical ML Pipeline ===")
    classic_ml = ClassicalMLPipeline(vector_size=100, model_type="svm")
    classic_ml.fit(df_train, label_names)
    
    print(f"\\n=== Phase 2: LLM Initialization ===")
    llm = LLMPipeline(label_names=label_names)
    
    uc = UnifiedClassifier(classic_ml, llm, cache_df)

    print(f"\\n=== Phase 3: Benchmarking F1 Score ===")
    # To respect API limits (15 RPM), we can use a stratified or random subset of the test data.
    # We sample if requested to save hours of processing.
    if subsample_size and subsample_size < len(df_test):
        test_sample = df_test.sample(n=subsample_size, random_state=42)
    else:
        test_sample = df_test
        
    print(f"Evaluating on {len(test_sample)} test samples...")
    y_true = []
    y_pred_classic = []
    y_pred_llm = []
    
    # Batch processing with LLM cache
    llm_preds, updated_cache = llm.predict_batch(test_sample['text'].tolist(), cache_df, save_every=20)
    uc.llm_cache = updated_cache # sync cache
    
    # Standardize predictions for metrics
    for i, (idx, row) in enumerate(test_sample.iterrows()):
        true_label = row['label_text']
        c_pred = classic_ml.predict([row['text']])[0]
        l_pred = llm_preds[i]
        
        y_true.append(true_label)
        y_pred_classic.append(c_pred)
        y_pred_llm.append(l_pred)
        
    f1_classic = f1_score(y_true, y_pred_classic, average='weighted', zero_division=0)
    f1_llm = f1_score(y_true, y_pred_llm, average='weighted', zero_division=0)
    
    print(f"-> Classical ML F1-Score: {f1_classic:.4f}")
    print(f"-> LLM F1-Score: {f1_llm:.4f}")

    print(f"\n=== Phase 4: Operational Benchmarking (Latency Test) ===")
    sample_10 = df_test.sample(n=10, random_state=42)
    
    latencies_classic = []
    latencies_llm = []
    
    # We avoid the cache for latency test by temporarily passing an empty cache DataFrame
    empty_cache = pd.DataFrame(columns=['text', 'predicted_label'])
    uc_latency = UnifiedClassifier(classic_ml, llm, empty_cache)
    
    for _, row in sample_10.iterrows():
        text = row['text']
        _, l_c = uc_latency.predict_intent(text, model_type="classical")
        
        # Real LLM Latency measurement
        _, l_l = uc_latency.predict_intent(text, model_type="llm")
        
        latencies_classic.append(l_c)
        latencies_llm.append(l_l)
        # Sleep to respect rate limits during live latency test
        time.sleep(1)
        
    avg_classic_lat = sum(latencies_classic)/len(latencies_classic)
    avg_llm_lat = sum(latencies_llm)/len(latencies_llm)
    
    print(f"Average Inference Latency (Classical SVM): {avg_classic_lat:.4f} seconds")
    print(f"Average Inference Latency (LLM Gemini Flash): {avg_llm_lat:.4f} seconds")

    print("\n=== Phase 5: Result Persistence ===")
    results_data = {
        "metrics": {
            "f1_classic": f1_classic,
            "f1_llm": f1_llm,
            "avg_latency_classic": avg_classic_lat,
            "avg_latency_llm": avg_llm_lat
        },
        "classification_samples": len(test_sample)
    }
    with open("results.json", "w") as f:
        import json
        json.dump(results_data, f, indent=4)
    print("Results saved to results.json.")

    print("\n=== Phase 6: Qualitative Error Analysis ===")
    # Let's find one example where Classical failed but LLM succeeded
    for i, (t_label, c_label, l_label) in enumerate(zip(y_true, y_pred_classic, y_pred_llm)):
        if c_label != t_label and l_label == t_label:
            print(f"\nNuanced Ticket (LLM got right, SVM failed):")
            print(f"Text: {test_sample.iloc[i]['text']}")
            print(f"True: {t_label} | SVM predicted: {c_label} | LLM predicted: {l_label}")
            break

if __name__ == "__main__":
    run_benchmarking(subsample_size=20)
