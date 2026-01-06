import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models
from transformers import DistilBertModel


# Config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 256
SEQ_LEN = 4

RESULT_DIR = "results/figures"
os.makedirs(RESULT_DIR, exist_ok=True)


# Visual Encoder

class VisualEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, embed_dim)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        feats = self.backbone(x).squeeze()
        feats = self.fc(feats)
        return feats.view(b, s, -1)


# Text Encoder

class TextEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, embed_dim)

        for p in self.bert.parameters():
            p.requires_grad = False

    def forward(self, ids, mask):
        b, s, l = ids.shape
        ids = ids.view(b * s, l)
        mask = mask.view(b * s, l)
        out = self.bert(ids, attention_mask=mask)
        cls = out.last_hidden_state[:, 0]
        feats = self.fc(cls)
        return feats.view(b, s, -1)


# Fusion + Temporal Model

class StoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisualEncoder(EMBED_DIM)
        self.text = TextEncoder(EMBED_DIM)
        self.fusion = nn.Linear(EMBED_DIM * 2, 512)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, imgs, ids, mask):
        v = self.visual(imgs)
        t = self.text(ids, mask)
        fused = torch.cat([v, t], dim=-1)
        fused = F.relu(self.fusion(fused))
        out, _ = self.lstm(fused)
        return v, t, out


# Run once (NO TRAINING)

if __name__ == "__main__":

    from utils import load_story_data

    dataset, loader = load_story_data()
    batch = next(iter(loader))

    model = StoryModel().to(DEVICE)
    model.eval()

    imgs = batch["input_images"].to(DEVICE)
    ids = batch["input_ids"].to(DEVICE)
    mask = batch["attention_mask"].to(DEVICE)

    with torch.no_grad():
        v_emb, t_emb, seq_out = model(imgs, ids, mask)

    
    # Plot 1: Embedding Magnitudes
    
    plt.figure()
    plt.plot(v_emb.mean(dim=-1).cpu()[0], label="Visual")
    plt.plot(t_emb.mean(dim=-1).cpu()[0], label="Text")
    plt.legend()
    plt.title("Embedding Magnitude per Step")
    plt.savefig(f"{RESULT_DIR}/embedding_magnitude.png")
    plt.close()

    
    # Plot 2: Temporal Feature Norm
    
    plt.figure()
    plt.plot(seq_out.norm(dim=-1).cpu()[0])
    plt.title("Temporal LSTM Feature Norm")
    plt.xlabel("Step")
    plt.ylabel("Norm")
    plt.savefig(f"{RESULT_DIR}/temporal_norm.png")
    plt.close()

    
    # Plot 3: Cosine Similarity
    
    sim = F.cosine_similarity(v_emb, t_emb, dim=-1)
    plt.figure()
    plt.bar(range(SEQ_LEN), sim.cpu()[0])
    plt.title("Visual–Text Cosine Similarity")
    plt.savefig(f"{RESULT_DIR}/cosine_similarity.png")
    plt.close()

    print("SUCCESS")
    print(v_emb.shape)
    print(t_emb.shape)
