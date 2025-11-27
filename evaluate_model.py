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
