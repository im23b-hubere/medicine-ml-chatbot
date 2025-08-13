import json
import random

def create_tiny_dataset():
    """Creates a very small version of the MedQuad dataset for Vercel deployment"""
    
    # Load original dataset
    with open('medquad_full.json', 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print(f"Original dataset size: {len(full_data)} entries")
    
    # Select a very small subset (200 entries for Vercel)
    sample_size = min(200, len(full_data))
    tiny_data = random.sample(full_data, sample_size)
    
    # Save tiny dataset
    with open('medquad_tiny.json', 'w', encoding='utf-8') as f:
        json.dump(tiny_data, f, ensure_ascii=False, indent=1)
    
    print(f"Created tiny dataset: {len(tiny_data)} entries")
    print(f"File size optimized for Vercel deployment")
    
    # Show some examples
    print("\nSample questions:")
    for i, item in enumerate(tiny_data[:3]):
        print(f"{i+1}. {item['question'][:80]}...")

if __name__ == "__main__":
    create_tiny_dataset()
