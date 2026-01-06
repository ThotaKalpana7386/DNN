import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt


# DIRECTORIES

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


# CONFIG (SAFE)

CONFIG = {
    "batch_size": 1,
    "seq_len": 4,
    "max_text_len": 32,
    "num_samples": 10   # STREAM ONLY FIRST 10
}


# DATASET CLASS

class StoryDataset(Dataset):
    def __init__(self, iterable_data, tokenizer, transform, seq_len):
        self.data = list(iterable_data)
        self.tokenizer = tokenizer
        self.transform = transform
        self.need = seq_len + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        images = item["images"]
        texts = item["story"]

        if isinstance(texts, str):
            texts = [texts]

        images += [images[-1]] * (self.need - len(images))
        texts += [texts[-1]] * (self.need - len(texts))

        images = images[-self.need:]
        texts = texts[-self.need:]

        img_tensor = torch.stack([
            self.transform(img.convert("RGB")) for img in images
        ])

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_text_len"],
            return_tensors="pt"
        )

        return {
            "input_images": img_tensor[:-1],
            "input_ids": enc["input_ids"][:-1],
            "attention_mask": enc["attention_mask"][:-1],
            "target_image": img_tensor[-1],
            "target_ids": enc["input_ids"][-1]
        }


# LOAD STREAMING DATASET (NO DISK)

def load_story_data():
    print("Streaming dataset from Hugging Face (NO DISK WRITE)...")

    stream = load_dataset(
        "daniel3303/StoryReasoning",
        split="train",
        streaming=True
    )

    # TAKE ONLY N SAMPLES
    samples = []
    for i, item in enumerate(stream):
        if i >= CONFIG["num_samples"]:
            break
        samples.append(item)

    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = StoryDataset(samples, tokenizer, transform, CONFIG["seq_len"])
    loader = DataLoader(dataset, batch_size=1)

    with open("results/tables/dataset_info.txt", "w") as f:
        f.write(f"Streamed samples: {len(dataset)}\n")

    return dataset, loader


# SAVE SAMPLE FIGURE

def save_sample(dataset):
    sample = dataset[0]
    imgs = sample["input_images"]

    plt.figure(figsize=(10, 3))
    for i in range(imgs.shape[0]):
        plt.subplot(1, imgs.shape[0], i + 1)
        plt.imshow(imgs[i].permute(1, 2, 0))
        plt.axis("off")

    path = "results/figures/sample_story.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved → {path}")


# MAIN

if __name__ == "__main__":
    dataset, _ = load_story_data()
    save_sample(dataset)

    sample = dataset[0]
    print("SUCCESS")
    print(sample["input_images"].shape)
    print(sample["input_ids"].shape)
