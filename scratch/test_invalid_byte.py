import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.tokenizer import BPETokenizer

def test():
    tokenizer = BPETokenizer.load(str(PROJECT_ROOT / "output"))
    # The emoji 😀 is 4 bytes: F0 9F 98 80
    # Let's find tokens that correspond to these bytes.
    # In ByteLevel, bytes are mapped to characters.
    # We can just encode an emoji and see if it splits into multiple tokens.
    ids = tokenizer.encode("😀")
    print(f"Tokens for 😀: {ids}")
    
    # Let's decode just the first token
    t1 = tokenizer.decode([ids[0]])
    print(f"Decoded token 1: {repr(t1)}")
    
    t12 = tokenizer.decode(ids[:2])
    print(f"Decoded token 1-2: {repr(t12)}")
    
    tall = tokenizer.decode(ids)
    print(f"Decoded all: {repr(tall)}")

if __name__ == "__main__":
    test()
