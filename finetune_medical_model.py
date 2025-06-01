import json
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# === Parameter ===
DATA_PATH = "medquad_full.json"           # Pfad zu deiner JSON-Datei
MODEL_NAME = "all-MiniLM-L6-v2"           # Pretrained Modell
OUTPUT_PATH = "medizin-embeddings-finetuned"  # Speicherort für das neue Modell
BATCH_SIZE = 16
EPOCHS = 2                                # Erhöhe für längeres Training

# === 1. Daten laden ===
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = [item['question'] for item in data]

# === 2. Trainingspaare generieren ===
train_examples = []

# Positive Paare: Frage mit sich selbst
for q in questions:
    train_examples.append(InputExample(texts=[q, q], label=1.0))

# Negative Paare: Frage mit zufälliger anderer Frage
for i in range(len(questions)):
    q1 = questions[i]
    q2 = random.choice(questions)
    if q1 != q2:
        train_examples.append(InputExample(texts=[q1, q2], label=0.0))

# Optional: Mehr negative Paare für bessere Balance
for _ in range(len(questions)):
    q1, q2 = random.sample(questions, 2)
    train_examples.append(InputExample(texts=[q1, q2], label=0.0))

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