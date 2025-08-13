from sentence_transformers import SentenceTransformer
import os

# Modell laden
model = SentenceTransformer('output/finetuned-st-medquad')

# Auf Hugging Face Hub hochladen
# Ersetze 'your-username' mit deinem HF Username
model.push_to_hub(
    "erichuber/finetuned-medical-model",
    private=False,  # Setze auf True für privates Repository
    commit_message="Fine-tuned medical model for semantic search"
)

print("Modell erfolgreich auf Hugging Face Hub hochgeladen!")
print("Verwende: model = SentenceTransformer('erichuber/finetuned-medical-model')")
