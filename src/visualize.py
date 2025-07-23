
# === File: src/visualize.py ===
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import REPORT_FOLDER, MODEL_NAMES, VISUALIZATION_FOLDER, PLOT_COLOR, PLOT_FIGSIZE, PLOT_ROTATION

def plot_model_scores(metric):
    scores = {}

    for model in MODEL_NAMES:
        report_path = os.path.join(REPORT_FOLDER, f"{model}_report.csv")
        if os.path.exists(report_path):
            df = pd.read_csv(report_path, index_col=0)
            if metric == "accuracy":
                if "accuracy" in df.index:
                    score = df.loc["accuracy"][0]  # accuracy is stored as a scalar row
                else:
                    score = None
            elif metric in df.index:
                score = df.loc[metric, 'weighted avg'] if 'weighted avg' in df.columns else df.loc['weighted avg', metric]
            else:
                score = df.loc['weighted avg', metric]
            if score is not None:
                scores[model] = score

    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    plt.figure(figsize=PLOT_FIGSIZE)
    plt.bar(scores.keys(), scores.values(), color=PLOT_COLOR)
    plt.ylabel(metric)
    plt.title(f"Model Comparison ({metric})")
    plt.xticks(rotation=PLOT_ROTATION)
    plt.tight_layout()
    save_path = os.path.join(VISUALIZATION_FOLDER, f"model_comparison_{metric}.png")
    plt.savefig(save_path)
    #plt.show()

def plot_combined_metrics(metrics):
    os.makedirs(VISUALIZATION_FOLDER, exist_ok=True)
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics), figsize=(len(metrics)*5, 5))
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        scores = {}
        for model in MODEL_NAMES:
            report_path = os.path.join(REPORT_FOLDER, f"{model}_report.csv")
            if os.path.exists(report_path):
                df = pd.read_csv(report_path, index_col=0)
                if metric == "accuracy":
                    if "accuracy" in df.index:
                        score = df.loc["accuracy"][0]
                    else:
                        score = None
                elif metric in df.index:
                    score = df.loc[metric, 'weighted avg'] if 'weighted avg' in df.columns else df.loc['weighted avg', metric]
                else:
                    score = df.loc['weighted avg', metric]
                if score is not None:
                    scores[model] = score

        axes[idx].bar(scores.keys(), scores.values(), color=PLOT_COLOR)
        axes[idx].set_title(f"{metric.title()}")
        axes[idx].tick_params(axis='x', rotation=PLOT_ROTATION)

    plt.tight_layout()
    save_path = os.path.join(VISUALIZATION_FOLDER, "model_comparison_all_metrics.png")
    plt.savefig(save_path)
    #plt.show()

def generate_all_visualizations():
    metrics = ["f1-score", "accuracy", "precision"]
    for metric in metrics:
        plot_model_scores(metric)
    plot_combined_metrics(metrics)

if __name__ == "__main__":
    generate_all_visualizations()
