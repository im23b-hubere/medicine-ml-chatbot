import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# === Parameter ===
DATA_PATH = "medquad_full.json"           # Pfad zu deiner JSON-Datei
MODEL_NAME = "all-MiniLM-L6-v2"           # Pretrained Modell
OUTPUT_PATH = "output/finetuned-st-medquad"  # Speicherort für das neue Modell
BATCH_SIZE = 16
EPOCHS = 2                                # Erhöhe für längeres Training

# === 1. Daten laden ===
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === 2. Trainingspaare generieren ===
train_examples = []

# Positive Paare: Frage und zugehörige Antwort
for item in data:
    if item.get("question") and item.get("answer"):
        train_examples.append(InputExample(texts=[item["question"], item["answer"]], label=1.0))

# Negative Paare: Frage und zufällige, nicht zugehörige Antwort
for item in data:
    if item.get("question"):
        wrong = random.choice(data)
        while wrong["answer"] == item["answer"]:
            wrong = random.choice(data)
        train_examples.append(InputExample(texts=[item["question"], wrong["answer"]], label=0.0))

print(f"Trainingsbeispiele: {len(train_examples)}")

# === 3. Modell laden ===
model = SentenceTransformer(MODEL_NAME)

# === 4. DataLoader und Loss ===
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.CosineSimilarityLoss(model)

# === 5. Training ===
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    show_progress_bar=True
)

# === 6. Modell speichern ===
model.save(OUTPUT_PATH)
print(f"Fine-tuned Modell gespeichert unter: {OUTPUT_PATH}")

print("\n\033[92mHinweis:\033[0m\nUm das Modell in web_terminal_chat.py zu verwenden, ersetze die Zeile\n  model = SentenceTransformer('all-MiniLM-L6-v2')\ndurch\n  model = SentenceTransformer('output/finetuned-st-medquad')\n") 