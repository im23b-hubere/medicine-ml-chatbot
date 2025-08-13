from sentence_transformers import SentenceTransformer
import torch
import os

# Modell laden
model = SentenceTransformer('output/finetuned-st-medquad')

# Modell optimieren für kleinere Größe
def optimize_model(model, output_path):
    # Quantisierung für kleinere Größe
    model.half()  # FP16 statt FP32
    
    # Modell speichern
    model.save(output_path)
    
    # Größe prüfen
    size_mb = sum(f.stat().st_size for f in os.scandir(output_path) if f.is_file()) / (1024*1024)
    print(f"Optimiertes Modell Größe: {size_mb:.1f}MB")

# Optimiertes Modell erstellen
optimize_model(model, 'output/optimized-medical-model')
