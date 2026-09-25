import sys
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.tokenizer import BPETokenizer
from src.generation import generate
from src.model.gpt import GPT, GPTConfig
import json

def test():
    device = "cpu"
    config_path = PROJECT_ROOT / "output" / "config.json"
    with open(config_path, "r") as f:
        config = GPTConfig.from_dict(json.load(f))
    model = GPT(config)
    state_dict = torch.load(PROJECT_ROOT / "output/pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    tokenizer = BPETokenizer.load(str(PROJECT_ROOT / "output"))
    
    prompt = "Write a poem on AI"
    prompt_ids = tokenizer.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    torch.manual_seed(1234)
    out = generate(model, idx, max_new_tokens=30, temperature=0.8, top_k=50, top_p=0.9, eos_id=tokenizer.endoftext_id)
    tokens = out[0].tolist()
    gen_ids = tokens[len(prompt_ids):]
    
    print(f"Decoded all at once: {repr(tokenizer.decode(tokens))}")
    print(f"Prompt + decoded gen: {repr(prompt + tokenizer.decode(gen_ids))}")
    
    # Also simulate the streaming decode
    prev = ""
    streamed = ""
    for i in range(1, len(gen_ids) + 1):
        curr = tokenizer.decode(gen_ids[:i])
        new_text = curr[len(prev):]
        streamed += new_text
        prev = curr
        
    print(f"Prompt + streamed: {repr(prompt + streamed)}")

if __name__ == "__main__":
    test()
