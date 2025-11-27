# ==============================================
# train_model.py
# Fine-tune XLM-RoBERTa on bilingual GBV dataset (multi-label)
# ==============================================

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

CATEGORIES = ["Physical_violence", "sexual_violence", "emotional_violence", "economic_violence"]


class GBVDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float),
        }


def train_gbv_classifier(
    train_file="data/processed/multilabel_training_data.csv",
    epochs=2,
    batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.001,
    dropout_rate=0.1,
):
    df = pd.read_csv(train_file)
    df.dropna(subset=["text"], inplace=True)
    df = df[df["text"].str.len() > 15]

    label_matrix = df[CATEGORIES].values.tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        label_matrix,
        test_size=0.2,
        random_state=42,
    )

    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(CATEGORIES),
        problem_type="multi_label_classification",
        hidden_dropout_prob=dropout_rate,
    )

    train_dataset = GBVDataset(train_texts, train_labels, tokenizer)
    val_dataset = GBVDataset(val_texts, val_labels, tokenizer)

    train_label_array = np.array(train_labels)
    pos_counts = train_label_array.sum(axis=0)
    total_samples = len(train_labels)
    pos_weight = (total_samples - pos_counts) / np.clip(pos_counts, a_min=1, a_max=None)
    class_weights = torch.tensor(pos_weight, dtype=torch.float)

    sample_weights = train_label_array.sum(axis=1)
    sample_weights = 1.0 / np.clip(sample_weights, a_min=1, a_max=None)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_metric = 0
    patience, patience_counter = 3, 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().astype(int).tolist())

        avg_val_loss = val_loss / len(val_loader)
        report = classification_report(
            all_labels, all_preds, target_names=CATEGORIES, zero_division=0, output_dict=True
        )
        micro_f1 = report["micro avg"]["f1-score"]

        print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}")
        print(f"             Val Loss={avg_val_loss:.4f}, Micro F1={micro_f1:.4f}")

        if micro_f1 > best_val_metric:
            best_val_metric = micro_f1
            patience_counter = 0
            model.save_pretrained("best_gbv_model")
            tokenizer.save_pretrained("best_gbv_model")
            print("Saved new best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nFinal Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CATEGORIES, zero_division=0))

    return model, tokenizer



if __name__ == "__main__":
    model, tokenizer = train_gbv_classifier()

    # Evaluation block: display precision, recall, and F1 for the best model on the validation set
    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import classification_report

    CATEGORIES = ["Physical_violence", "sexual_violence", "emotional_violence", "economic_violence"]
    df = pd.read_csv('data/processed/multilabel_training_data.csv')
    tokenizer = AutoTokenizer.from_pretrained('best_gbv_model')
    model = AutoModelForSequenceClassification.from_pretrained('best_gbv_model').eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for chunk in range(0, len(df), 64):
            batch = df.iloc[chunk:chunk+64]
            enc = tokenizer(batch['text'].tolist(), padding=True, truncation=True, max_length=256, return_tensors='pt').to(device)
            logits = model(**enc).logits
            preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int).tolist()
            all_preds.extend(preds)
            all_labels.extend(batch[CATEGORIES].values.astype(int).tolist())

    print(classification_report(all_labels, all_preds, target_names=CATEGORIES, zero_division=0))
