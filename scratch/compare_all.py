import sys
from pathlib import Path
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
    
    prompt = "Write a poem on AI"
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [tokenizer.endoftext_id]
        
    for seed in range(5):
        print(f"--- Seed {seed} ---")
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        torch.manual_seed(seed)
        out = generate(model, idx, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.9, eos_id=tokenizer.endoftext_id)
        tokens1 = out[0].tolist()
        
        idx2 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        torch.manual_seed(seed)
        captured = []
        original_decode = tokenizer.decode
        def mock_decode(ids, skip=True):
            if ids: captured.append(ids[-1])
            return original_decode(ids, skip)
        tokenizer.decode = mock_decode
        
        stream_out = list(generate_stream(model, idx2, tokenizer, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.9, eos_id=tokenizer.endoftext_id))
        tokenizer.decode = original_decode
        
        tokens2 = prompt_ids + captured
        
        t1_str = tokenizer.decode(tokens1)
        t2_str = tokenizer.decode(tokens2)
        
        print(f"generate: {t1_str}")
        print(f"stream  : {t2_str}")
        print(f"Match: {tokens1 == tokens2}")
        print()

if __name__ == "__main__":
    test()
