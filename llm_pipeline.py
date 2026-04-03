import os
import time
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google import genai
from google.genai import errors
from data_loader import get_cached_prediction, add_prediction_to_cache, save_llm_cache

class LLMPipeline:
    def __init__(self, api_key=None, label_names=None):
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.label_names = label_names

    def build_prompt(self, ticket_text):
        """Constructs the prompt for zero-shot or few-shot classification."""
        labels_str = "\\n".join([f"- {label}" for label in self.label_names])
        prompt = f"""You are a customer service intent classification AI for a bank.
Your task is to classify the USER QUERY into exactly one of the following official categories.
Do not add any conversational text. Reply ONLY with the exact category name.

Categories:
{labels_str}

USER QUERY: "{ticket_text}"
CATEGORY:"""
        return prompt

    @retry(
        retry=retry_if_exception_type(errors.APIError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(10)
    )
    def query_gemini_with_retry(self, prompt):
        """Calls Gemini API with resilient exponential backoff for 429 Rate Limit Errors."""
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text.strip()

    def process_ticket(self, text, cache_df):
        """Processes a single ticket: checks cache first, else queries API."""
        cached_label = get_cached_prediction(text, cache_df)
        if cached_label is not None:
            return cached_label, cache_df, True # True means it was cached

        prompt = self.build_prompt(text)
        try:
            prediction = self.query_gemini_with_retry(prompt)
        except Exception as e:
            print(f"Failed to process ticket after retries due to: {e}")
            prediction = "ERROR"
            
        # Optional: try to match the prediction exactly to label_names
        # Sometimes the LLM might add a period or slight variation.
        for label in self.label_names:
            if label.lower() == prediction.lower().strip("."):
                prediction = label
                break
                
        # Add to cache dataframe
        cache_df = add_prediction_to_cache(text, prediction, cache_df)
        return prediction, cache_df, False # False means it was newly queried

    def predict_batch(self, texts, cache_df, max_queries=None, save_every=20):
        """
        Predicts a batch of texts.
        To avoid high waiting times, you can set max_queries to limit how many 
        new API calls are made in one run.
        """
        predictions = []
        new_queries = 0
        
        for i, text in enumerate(texts):
            pred, cache_df, was_cached = self.process_ticket(text, cache_df)
            predictions.append(pred)
            
            if not was_cached:
                new_queries += 1
                # Save cache periodically
                if new_queries % save_every == 0:
                    save_llm_cache(cache_df)
                    print(f"API Progress: {new_queries} queries made. Cache saved.")
                
                # Sleep to be nice to the rate limit even if tenacity handles errors
                time.sleep(2)  # For 15 RPM, roughly 4 seconds per request. 2s is a minimal buffer.
                
            if max_queries and new_queries >= max_queries:
                print(f"Reached max queries ({max_queries}). Breaking early.")
                # We extend the rest with 'UNPROCESSED' just to pad the list
                remaining = len(texts) - len(predictions)
                predictions.extend(["UNPROCESSED"] * remaining)
                break
                
        # Final save
        save_llm_cache(cache_df)
        return predictions, cache_df
