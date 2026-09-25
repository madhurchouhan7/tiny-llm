import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.tokenizer import BPETokenizer

def test():
    tokenizer = BPETokenizer.load(str(PROJECT_ROOT / "output"))
    prompt = "Write a poem on AI"
    prompt_ids = tokenizer.encode(prompt)
    
    gen_ids = [407, 38, 407, 38]
    
    print(f"Prompt: {prompt}")
    print(f"Decoded prompt_ids: {tokenizer.decode(prompt_ids)}")
    print(f"Decoded prompt_ids + gen_ids: {tokenizer.decode(prompt_ids + gen_ids)}")
    print(f"Prompt + decoded gen_ids: {prompt + tokenizer.decode(gen_ids)}")

if __name__ == "__main__":
    test()
