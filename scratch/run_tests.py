import sys
from pathlib import Path
import time
import torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.tokenizer import BPETokenizer
from src.generation import generate, generate_stream
from src.model.gpt import GPT, GPTConfig
import json

def test():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config_path = PROJECT_ROOT / "output" / "config.json"
    with open(config_path, "r") as f:
        config = GPTConfig.from_dict(json.load(f))
    model = GPT(config)
    state_dict = torch.load(PROJECT_ROOT / "output/pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    tokenizer = BPETokenizer.load(str(PROJECT_ROOT / "output"))
    
    prompt = "The future of artificial intelligence"
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids: prompt_ids = [tokenizer.endoftext_id]
    
    # Test 1
    print("="*48)
    print("TEST 1 - NORMAL GENERATION")
    print("="*48)
    idx1 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    torch.manual_seed(999)
    out1 = generate(model, idx1, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.9, eos_id=tokenizer.endoftext_id)
    tokens1 = out1[0, len(prompt_ids):].tolist()
    print(prompt + tokenizer.decode(tokens1))
    
    print("\n" + "="*48)
    print("TEST 2 - STREAMING GENERATION")
    print("="*48)
    idx2 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    torch.manual_seed(999)
    print(prompt, end="", flush=True)
    for token_text in generate_stream(model, idx2, tokenizer, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.9, eos_id=tokenizer.endoftext_id):
        print(token_text, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    test()
