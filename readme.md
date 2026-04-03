# Comparative Analysis of NLP Architectures for Automated Intent Classification

## 1. Executive Summary
This project presents a rigorous comparison between two distinct Natural Language Processing (NLP) architectures for the intent classification of customer service banking queries. Using the `MTEB/Banking77` dataset, which spans 77 fine-grained intent categories, we evaluate a **Classical Machine Learning Pipeline** (Word2Vec Embeddings + Support Vector Machine) against a **Modern Generative AI Pipeline** (Gemini-2.5-Flash Zero-/Few-Shot Prompting).

The objective is to provide a clear Business Case on the tradeoffs between operational latency, performance (F1-Score), development complexity, and cost when classifying real-world banking queries at scale.

---

## 2. Methodology & Implementation

### 2.1 The Classical ML Pipeline
**Architecture:** `Word2Vec` + `Support Vector Machine (SVM)`

- **Data Engineering:** The text was normalized through lowercasing and punctuation removal. 
- **Feature Extraction:** A continuous Skip-Gram/CBOW `Word2Vec` model was trained directly on the `Banking77` training set. Sentence representations were constructed by calculating the mean vector of the tokens in the query.
- **Classification:** An SVM classifier with a linear kernel was trained on the dense vector representations.
- **Optimization Strategy:** The pipeline operates purely offline and requires zero ongoing API calls. Sentence embeddings provide a rapid and lightweight heuristic for semantic similarity.

### 2.2 The LLM Pipeline (Gemini-2.5-Flash)
**Architecture:** `Gemini API` + `Tenacity Resilient Orchestration` + `Parquet Caching`

- **Embedding & Classification:** Using prompt engineering, the Gemini-2.5-Flash model was queried to classify the text directly. The system prompt contained the exact banking categories to constrain the output generation.
- **Resilient Orchestration (Error 429 Handling):** Public API rate limits (15 RPM) inherently cause `HTTP 429` Rate Limit Errors. A robust exponential backoff strategy was implemented via the `tenacity` library, ensuring fault tolerance and uninterrupted batch execution.
- **Local Persistence & Caching:** To avoid redundant API calls and save computational time, query predictions were hashed and persisted locally into `pyarrow`-backed Parquet files. If a ticket was queried previously, the prediction was instantly loaded from the cache.

---

## 3. Benchmarking & Comparative Evaluation

### 3.1 Quantitative Performance (Test Subset)
*Note: Due to API rate limits, the full evaluation was sampled across a representative test set of 20 tickets.*

- **Classical ML (SVM) F1-Score:** `0.2500` (Averaged Word2Vec embeddings lack contextual nuance)
- **Modern LLM (Gemini) F1-Score:** `1.0000` (Excellent zero-shot generalization capabilities)

### 3.2 Operational Benchmarking (Latency)
A dedicated latency test was run on a random batch of 10 incoming tickets to simulate a live customer service environment.
- **Classical ML Inference Latency:** `0.0030` seconds/query
- **LLM Inference Latency (Network bounded):** `1.49` seconds/query

### 3.3 Complexity Audit & Qualitative Analysis
**Complexity:**
- **Classical ML:** High upfront implementation complexity (Training loops, feature scaling, model saving/loading). However, operationally cost-effective.
- **LLM Pipeline:** Lower algorithmic complexity, but high architectural complexity involving network orchestration, cache state synchronization, and prompt engineering constraints.

**Error Analysis (Sarcasm & Nuance):**
The LLM inherently understands out-of-distribution nuances better. Ambiguous, short questions such as *"How do I link this new card?"* confuse the SVM (which erroneously predicted `cancel_transfer` due to weak vector space separation) whereas the LLM effectively associates the entire semantic clause with `card_linking`.

---

## 4. Business & Production Case Study

As the Lead Data Scientist for this marketplace, scaling to **1 Million requests per month** requires evaluating the intersection of variable costs and performance. 

### 4.1 Cost Estimation (1M Requests)

**1. Classical ML Architecture:**
- **Infrastructure:** Hosting on a `c4-standard-8` (GCP).
- **Compute Cost:** ~ $300 - $400 / month (always-on cloud VM).
- **API Cost:** $0.
- **Total:** ~ $350 / month perfectly predictable cost.

**2. Modern LLM Architecture:**
- **API Cost:** Gemini 2.5 Flash costs ≈ $0.075 per 1M input tokens. At ~40 tokens per query * 1M queries = 40M tokens. Base API cost ≈ $3.00.
- **Infrastructure:** A smaller orchestration server (`c4-standard-2` GCP) is required. Cost: ~$85 / month.
- **Total:** ~ $90 / month. 

### 4.2 Executive Decision
**Recommendation: Hybrid LLM-Assisted Classification with Heavy Caching**

While traditional intuition suggests the Classical ML pipeline is cheaper, Gemini-2.5-Flash disrupts this paradigm. The variable API cost for 1 Million queries is remarkably low ($3.00), making the overall infrastructure (orchestration on a smaller server) far cheaper than dedicating an 8-core machine to SVM vectorization.

Furthermore, the LLM provides intrinsically higher adaptability to new banking categories without requiring full retraining. Therefore, **I recommend deploying the LLM Architecture** bolstered by the implemented `PyArrow`/Parquet caching layer.

The latency trade-off (LLM API overhead vs instantaneous local SVM) is acceptable for asynchronous ticket-routing systems, provided a strict SLA for synchronous live-chat does not demand sub-100ms response times. If real-time inference becomes mandatory, the architecture can fall back on the locally cached embeddings or a distilled baseline model.

---

## 5. How to Run

### 5.1 Prerequisites
- Python 3.10+
- A Google Gemini API Key (stored in `.env` as `GEMINI_API_KEY`)

### 5.2 Installation
```bash
pip install -r requirements.txt
```

### 5.3 Execution
1. **Run the Benchmark**:
   ```bash
   python benchmarking.py
   ```
   This will train the SVM, query Gemini (with caching), and save metrics to `results.json`.

2. **Generate Visualizations**:
   ```bash
   python visualize_results.py
   ```
   This generates `f1_score_comparison.png` and `latency_comparison.png`.

---

## 6. Project structure
- `classical_ml.py`: Word2Vec + SVM Implementation.
- `llm_pipeline.py`: Gemini-1.5-Flash implementation with exponential backoff.
- `data_loader.py`: Hugging Face dataset integration and Parquet caching logic.
- `benchmarking.py`: Unified entry point for comparative evaluation.
- `visualize_results.py`: Results charting utility.
