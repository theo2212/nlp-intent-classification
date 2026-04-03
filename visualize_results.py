import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_plots(results_file="results.json"):
    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Run benchmarking.py first.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    metrics = data["metrics"]
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. F1-Score Comparison
    plt.figure(figsize=(8, 6))
    models = ["Classical ML (SVM)", "Modern LLM (Gemini)"]
    f1_scores = [metrics["f1_classic"], metrics["f1_llm"]]
    
    colors = ["#4C72B0", "#55A868"]
    sns.barplot(x=models, y=f1_scores, palette=colors)
    plt.title(f"Accuracy Comparison (Weighted F1-Score)\nSample Size: {data['classification_samples']}", fontsize=14)
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.1)
    
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("f1_score_comparison.png")
    print("Saved f1_score_comparison.png")

    # 2. Latency Comparison (Log Scale)
    plt.figure(figsize=(8, 6))
    latencies = [metrics["avg_latency_classic"], metrics["avg_latency_llm"]]
    
    ax = sns.barplot(x=models, y=latencies, palette=colors)
    ax.set_yscale("log")
    plt.title("Operational Benchmarking: Inference Latency", fontsize=14)
    plt.ylabel("Avg Latency (seconds) - Log Scale", fontsize=12)
    
    for i, v in enumerate(latencies):
        plt.text(i, v * 1.1, f"{v:.4f}s", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("latency_comparison.png")
    print("Saved latency_comparison.png")

if __name__ == "__main__":
    generate_plots()
