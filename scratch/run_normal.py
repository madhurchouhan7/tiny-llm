import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT, GPTConfig
from src.tokenizer import BPETokenizer
from src.generation import generate, generate_text
import json

def test():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    config_path = PROJECT_ROOT / "output" / "config.json"
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = GPTConfig.from_dict(config_dict)
    
    model = GPT(config)
    
    weights_path = PROJECT_ROOT / "output" / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    tokenizer = BPETokenizer.load(str(PROJECT_ROOT / "output"))
    
    prompt = "The future of artificial intelligence"
    print(generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        device=device
    ))

if __name__ == "__main__":
    test()
