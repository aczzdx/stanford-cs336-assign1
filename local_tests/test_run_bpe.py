import pytest as pt
from pathlib import Path
import pprint as pp

from cs336_basics.bpe import train_bpe

def test_bpe_small(small_corpus_path: Path):
    
    vocab, merges = train_bpe(
        num_processes=1,
        num_chunks=1,
        vocab_size=(1 + 256 + 6),
        special_tokens=["<|endoftext|>"],
        data_file_path=small_corpus_path.absolute(),
        is_debug=True
    )
    
    pp.pprint(f"len(vocab) = {len(vocab)}")
    pp.pprint({k: v for k,v in vocab.items() if k >= 256})
    pp.pprint(merges)