import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.tokenizer import BPETokenizer

def test():
    tokenizer = BPETokenizer.load(str(PROJECT_ROOT / "output"))
    # find token for " WE"
    ids = tokenizer.encode("WE WE WE")
    print(f"Encoded 'WE WE WE': {ids}")
    for id in ids:
        print(f"Token {id} -> '{tokenizer.decode([id])}'")
        
    print(f"Decode [ids[0]]: '{tokenizer.decode([ids[0]])}'")
    print(f"Decode [ids[0], ids[1]]: '{tokenizer.decode([ids[0], ids[1]])}'")
    
    current = tokenizer.decode([ids[0], ids[1]])
    prev = tokenizer.decode([ids[0]])
    new = current[len(prev):]
    print(f"prev: '{prev}', current: '{current}', new: '{new}'")

if __name__ == "__main__":
    test()
