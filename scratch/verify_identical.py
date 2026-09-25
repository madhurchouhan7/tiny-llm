import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.gpt import GPT, GPTConfig
from src.tokenizer import BPETokenizer
from src.generation import generate, generate_stream
import json

def main():
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
    prompt_ids = tokenizer.encode(prompt)
    if len(prompt_ids) == 0:
        prompt_ids = [tokenizer.endoftext_id]
        
    idx1 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    idx2 = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    
    torch.manual_seed(42)
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
    
    torch.manual_seed(42)
    # Patch generate_stream locally to return the next_ids instead of text?
    # No, we can just intercept the `generated_ids` variable inside generate_stream.
    # Wait, python generators don't let us easily inspect locals.
    # I'll just temporarily monkeypatch tokenizer.decode to capture the IDs!
    captured_ids = []
    original_decode = tokenizer.decode
    def mock_decode(ids, skip_special_tokens=True):
        if ids:
            captured_ids.append(ids[-1])
        return original_decode(ids, skip_special_tokens)
    tokenizer.decode = mock_decode
    
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
    tokenizer.decode = original_decode
    
    tokens2 = captured_ids
    
    print(f"Match: {tokens1 == tokens2}")
    if tokens1 != tokens2:
        print(f"Normal tokens: {tokens1}")
        print(f"Stream tokens: {tokens2}")
        for i, (t1, t2) in enumerate(zip(tokens1, tokens2)):
            if t1 != t2:
                print(f"Mismatch at step {i}: {t1} != {t2}")
                break

if __name__ == "__main__":
    main()
