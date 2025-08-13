import json
import random

def create_small_dataset():
    """Creates a smaller version of the MedQuad dataset for Vercel deployment"""
    
    # Load original dataset
    with open('medquad_full.json', 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    print(f"Original dataset size: {len(full_data)} entries")
    
    # Select a random subset (1000 entries for demo)
    sample_size = min(1000, len(full_data))
    small_data = random.sample(full_data, sample_size)
    
    # Save smaller dataset
    with open('medquad_small.json', 'w', encoding='utf-8') as f:
        json.dump(small_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created smaller dataset: {len(small_data)} entries")
    print(f"File size reduced for Vercel deployment")
    
    # Show some examples
    print("\nSample questions:")
    for i, item in enumerate(small_data[:3]):
        print(f"{i+1}. {item['question'][:100]}...")

if __name__ == "__main__":
    create_small_dataset()
