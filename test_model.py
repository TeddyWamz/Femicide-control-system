import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# 1️⃣ Load tokenizer and trained model
MODEL_PATH = "best_gbv_model"
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_PATH)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 2️⃣ Define the label mapping (must match your training script)
id2label = {
    0: "Physical_violence",
    1: "sexual_violence",
    2: "emotional_violence",
    3: "economic_violence"
}

# 3️⃣ Create a function to make predictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    label = id2label[predicted_class_id]
    return label

# 4️⃣ Test with English and Swahili examples
examples = [
    "He slapped me and broke my phone.",  # Physical
    "Alinilazimisha kufanya ngono bila ridhaa yangu.",  # Sexual
    "Ananidharau na kunikemea kila siku mbele ya watu.",  # Emotional
    "Ananinyima pesa za matumizi kila wakati.",  # Economic
    "He insulted me in front of my colleagues.",  # Emotional
    "She controls all the money and doesn’t let me buy anything.",  # Economic
    "Alinipiga na kuniumiza.",  # Physical
    "Ananichapa kila siku.",  # Physical
    "He locked me in the house and refused to let me out.",  # Physical
    "He insulted me and called me names.",  # Emotional
    "Ananinyima chakula kama sitimtii mahitaji yake.",  # Economic
    "alinipoint na kisu.", # Emotional
    "He controls all the money, won't allow me to work, and takes away my bank cards.",
    "I was beaten by my partner na aliniumiza sana.",
    "Amekataa kunipa pesa za chakula na mavazi.",
    "alinitusi na kuniita majina kwa sababu chakula kilikuwa baridi.",
    "Alinipa dawa zilizofanya nilale alfu akafanya ngono nami bila ridhaa yangu."

    
    
]

for text in examples:
    print(f"\n🗣️ Text: {text}")
    print(f"🔎 Predicted category: {predict(text)}")
