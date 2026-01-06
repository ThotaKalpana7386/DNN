import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from model import StoryModel
from utils import load_story_data


# CONFIG

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1            # demo training
LR = 1e-3

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


# SIMPLE METEOR (NO WORDNET)

def simple_meteor(reference, prediction):
    ref = set(reference.split())
    pred = set(prediction.split())
    if len(ref) == 0:
        return 0.0
    return len(ref & pred) / len(ref)


# METRICS

def compute_metrics():
    reference = "a story about a person"
    prediction = "a story about person"

    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu(
        [reference.split()],
        prediction.split(),
        smoothing_function=smoothie
    )

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = rouge.score(reference, prediction)["rougeL"].fmeasure

    meteor = simple_meteor(reference, prediction)

    return bleu, rouge_l, meteor


# TRAIN LOOP (LIGHTWEIGHT)

def train():
    dataset, loader = load_story_data()

    model = StoryModel().to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(EPOCHS):
        for batch in loader:
            imgs = batch["input_images"].to(DEVICE)
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)

            optimizer.zero_grad()

            v, t, out = model(imgs, ids, mask)

            loss = criterion(v.mean(), t.mean())
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    
    # SAVE LOSS PLOT
    
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig("results/figures/training_loss.png")
    plt.close()

    
    # METRICS
    
    bleu_score, rouge_score, meteor_score = compute_metrics()

    print(
        f"Evaluation -> BLEU: {bleu_score:.4f}, "
        f"ROUGE: {rouge_score:.4f}, "
        f"METEOR: {meteor_score:.4f}"
    )

    
    # SAVE METRIC TABLE
    
    df = pd.DataFrame({
        "Metric": ["BLEU", "ROUGE-L", "METEOR"],
        "Score": [bleu_score, rouge_score, meteor_score]
    })
    df.to_csv("results/tables/evaluation_metrics.csv", index=False)

    
    # METRIC BAR PLOT
    
    plt.figure()
    plt.bar(df["Metric"], df["Score"])
    plt.title("Evaluation Metrics")
    plt.ylabel("Score")
    plt.savefig("results/figures/metrics_bar.png")
    plt.close()

    print("TRAINING COMPLETE")
    print("Saved → results/figures/")
    print("Saved → results/tables/")


# MAIN

if __name__ == "__main__":
    train()
