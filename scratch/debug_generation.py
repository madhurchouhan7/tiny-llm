import sys
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT, GPTConfig
from src.tokenizer import BPETokenizer
from src.generation import generate, generate_stream
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
    
    prompt = "Write a poem on AI"
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.endoftext_id]
        
    idx1 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    idx2 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    # Run original generate
    torch.manual_seed(1337)
    out1 = generate(
        model=model,
        idx=idx1,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        eos_id=tokenizer.endoftext_id,
    )
    tokens1 = out1[0, len(prompt_ids):].tolist()
    
    # Run stream
    torch.manual_seed(1337)
    streamed_texts = list(generate_stream(
        model=model,
        idx=idx2,
        tokenizer=tokenizer,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        eos_id=tokenizer.endoftext_id,
    ))
    
    text1 = tokenizer.decode(tokens1, skip_special_tokens=False)
    text2 = "".join(streamed_texts)
    
    print(f"Normal text: {repr(text1)}")
    print(f"Stream text: {repr(text2)}")

if __name__ == "__main__":
    test()
