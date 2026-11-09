import multiprocessing as mp
import os
import time
import regex as re
from typing import Any, BinaryIO, Iterable, TypedDict, Literal, DefaultDict
from collections import defaultdict
# import tqdm based on different platform
from tqdm.std import trange, tqdm
import click
import pickle as pkl
import itertools
import pprint as pp


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_REGEX = re.compile(PAT)
FREQ_TABLE_PKL_FILEPATH = "data/freq_table.pkl"

END_OF_TEXT_TOKEN = "<|endoftext|>"

FrequencyTableDefualtDict = DefaultDict[bytes, int]
FrequencyTable = dict[bytes, int]

VocabTable = dict[int, bytes]
MergeRecord = tuple[bytes, bytes]
MergeRecordList = list[MergeRecord]

PairFreqDefaultDict = DefaultDict[tuple[bytes, bytes], int]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    for split_special_token in split_special_tokens:
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in trange(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the first special token in the mini chunk
            first_found_at = None
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at >= 0 and (first_found_at is None or first_found_at > found_at):
                    first_found_at = found_at
            
            if first_found_at is not None:
                chunk_boundaries[bi] = initial_position + first_found_at
                break
            initial_position += mini_chunk_size
            

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(idx: int, start: int, end: int, data_file_path: str, special_tokens: list[str]) -> FrequencyTableDefualtDict:
    
    pattern_split_special_token = re.compile(r'\L<tokens>', tokens=special_tokens)
    ret = defaultdict(int)
    with open(data_file_path, "rb") as f:       
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        doc_match_iterable: Iterable
        if special_tokens:
            doc_match_iterable = pattern_split_special_token.splititer(chunk)
        else:
            doc_match_iterable = [chunk]

        for doc_match in doc_match_iterable:
            if len(doc_match) == 0:
                continue
            for m in PAT_REGEX.finditer(doc_match):
                ret[m.group(0).encode('utf-8')] += 1

    return ret

def wrapped_pretokenize(args): return pretokenize(*args)

def init_vocab(special_tokens: list[str]) -> dict[int, bytes]:
    ret: dict[int, bytes] = {}

    for idx in range(1 << 8):
        b = idx.to_bytes()
        ret[idx] = b

    for i, special_token in enumerate(special_tokens):
        ret[i + (1 << 8)] = special_token.encode('utf-8')
    return ret

def merge_frequency_table(lhs: FrequencyTableDefualtDict, rhs: FrequencyTable) -> FrequencyTableDefualtDict:

    for subtoken, freq in rhs.items():
        lhs[subtoken] += freq
    return lhs

def train_bpe(
    num_processes: int,
    data_file_path: str | os.PathLike,
    special_tokens: list[str],
    num_chunks: int,
    vocab_size: int,    
    num_chunks_limit: int | None = None,
    will_dump_freq_table_pkl: bool = False,
    will_load_freq_table_pkl: bool = False,
    is_debug: bool = False
):
    freq_table: FrequencyTableDefualtDict = get_freq_table(
        num_processes, data_file_path, special_tokens, num_chunks, num_chunks_limit, will_dump_freq_table_pkl, will_load_freq_table_pkl
    )
    
    if is_debug:
        for subtoken, freq in itertools.islice(freq_table.items(), 10):
            click.echo(f"'{subtoken}' -- {freq}")
    
    vocab = init_vocab(special_tokens)
    subtoken_to_token_list: dict[bytes, list[bytes]] = dict()

    for subtoken in freq_table.keys():
        subtoken_to_token_list[subtoken] = [x.to_bytes() for x in subtoken]
    
    num_init_vocab = len(vocab)
    num_remaining_vocab = vocab_size - len(vocab)

    merges: MergeRecordList = []
    current_pair_freq: PairFreqDefaultDict | None = None
    for i in range(num_remaining_vocab):
        # pp.pprint(current_pair_freq)
        current_pair_freq = find_most_freq_and_mergs(
            vocab=vocab,
            merges=merges,
            subtoken_to_token_list=subtoken_to_token_list,
            subtoken_freq_table = freq_table,
            previous_pairs_freq=current_pair_freq,
            new_idx=num_init_vocab + i,
        )
    
    return vocab, merges

def find_most_freq_and_mergs(
    vocab: VocabTable,
    merges: MergeRecordList,
    subtoken_to_token_list: dict[bytes, list[bytes]],
    subtoken_freq_table: dict[bytes, int],
    previous_pairs_freq: PairFreqDefaultDict | None,
    new_idx: int
):
    if previous_pairs_freq is None:
        pairs_freq = count_pair_freq(subtoken_to_token_list, subtoken_freq_table)
    else:
        pairs_freq = previous_pairs_freq
    
    new_token_freq, pair_to_merge = find_pair_to_merge(pairs_freq)
    
    merges.append(pair_to_merge)
    new_token = pair_to_merge[0] + pair_to_merge[1]
    vocab[new_idx] = new_token
    
    merge_new_token_pair_to_single(
        subtoken_to_token_list=subtoken_to_token_list,
        subtoken_freq_table=subtoken_freq_table,
        pair_to_merge=pair_to_merge,
        pair_freq_table=pairs_freq,
        new_token=new_token
    )
    
    return pairs_freq
    

def merge_new_token_pair_to_single(
    subtoken_to_token_list,
    subtoken_freq_table,
    pair_to_merge,
    pair_freq_table: PairFreqDefaultDict,
    new_token
):

    for subtoken, subtoken_freq in subtoken_freq_table.items():
        # PRECONDITION: len(new_token) >= 2
        if new_token in subtoken:
            new_token_list = []
            old_token_list = subtoken_to_token_list[subtoken]

            for token_idx, token in enumerate(old_token_list):
                new_token_list.append(token)
                
                if len(new_token_list) >= 2:
                    w = tuple(new_token_list[-2:])
                    if w == pair_to_merge:
                        new_token_list.pop()
                        new_token_list.pop()
                        new_token_list.append(new_token)
                        pair_freq_table[pair_to_merge] -= subtoken_freq
                        
                        # (prev?, fst, sec, next?) -> (prev?, fst-sec, next?)
                        if len(new_token_list) >= 2:
                            # (prev, fst, sec)
                            old_pair = (new_token_list[-2], pair_to_merge[0])
                            if old_pair in pair_freq_table:
                                pair_freq_table[old_pair] -= subtoken_freq
                                if (pair_freq_table[old_pair] == 0):
                                    del pair_freq_table[old_pair]
                            pair_freq_table[(new_token_list[-2], new_token)] += subtoken_freq

                        if token_idx < len(old_token_list) - 1:
                            # (fst, sec, next)
                            old_pair = (pair_to_merge[1], old_token_list[token_idx + 1])
                            if old_pair in pair_freq_table:
                                pair_freq_table[old_pair] -= subtoken_freq
                                if (pair_freq_table[old_pair] == 0):
                                    del pair_freq_table[old_pair]
                            pair_freq_table[(new_token, old_token_list[token_idx + 1])] += subtoken_freq

            subtoken_to_token_list[subtoken] = new_token_list
    
    assert pair_freq_table[pair_to_merge] == 0
    del pair_freq_table[pair_to_merge]


def find_pair_to_merge(pairs_freq: PairFreqDefaultDict):
    max_freq = -1
    max_freq_candidates: list[MergeRecord] = []
    for p, freq in pairs_freq.items():
        if freq > max_freq:
            max_freq = freq
            max_freq_candidates = [p]
        elif freq == max_freq:
            max_freq_candidates.append(p)
    pair_to_merge = max(max_freq_candidates)
    return max_freq, pair_to_merge
    
    

def count_pair_freq(subtoken_to_token_list, subtoken_freq_table):
    pairs_freq: PairFreqDefaultDict = defaultdict(int)
    
    for subtoken, freq in subtoken_freq_table.items():
        subtoken_token_list = subtoken_to_token_list[subtoken]
        for lhs, rhs in zip(subtoken_token_list[:-1], subtoken_token_list[1:]):
            pairs_freq[(lhs, rhs)] += freq
    return pairs_freq
    
    
        
    

def get_freq_table(
    num_processes, data_file_path, special_tokens, num_chunks, num_chunks_limit, will_dump_freq_table_pkl, will_load_freq_table_pkl
) -> FrequencyTableDefualtDict:
    if will_load_freq_table_pkl:
        click.echo("loading from pkl")
        with open(FREQ_TABLE_PKL_FILEPATH, "rb") as f:
            ret: FrequencyTableDefualtDict = pkl.load(f)
        return ret

    split_special_tokens = [x.encode('utf-8') for x in special_tokens]
    with open(data_file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, split_special_tokens)

    freq_table: FrequencyTableDefualtDict = defaultdict(int)

    params = [
        (i, start_end_pair[0], start_end_pair[1], data_file_path, special_tokens)
        for i, start_end_pair in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    
    if num_chunks_limit is not None:
        click.echo(f"found num_chunks_limit = {num_chunks_limit}, only use the first {num_chunks_limit} chunk(s)")
        params = params[:num_chunks_limit]

    with mp.Pool(num_processes) as pool:
        for partial_freq_table in tqdm(
            pool.imap_unordered(wrapped_pretokenize, params),
            total=len(params)
        ):
            merge_frequency_table(freq_table, partial_freq_table)

    # Stats of freq_table
    click.echo(f"Number of chucks: {len(params)}")
    click.echo(f"Number of sub-tokens: {len(freq_table.keys())}")
    
    # Dumps the ouput 
    if will_dump_freq_table_pkl:
        with open(FREQ_TABLE_PKL_FILEPATH, "wb") as f:
            pkl.dump(freq_table, f)
    
    click.echo(f"freq_table constructed")
    return freq_table
        
        
@click.command()
@click.option('--num-processes', default=4)
@click.option('--data-file', 'data_file_path', type=click.Path(exists=True))
@click.option('--num-chunks', default=10_000)
@click.option('--num-chunks-limit', default=None, type=int)
@click.option('--vocab-size', default=10_000)
def run_train_bpe(
    num_processes: int,
    data_file_path: str,
    num_chunks: int,
    num_chunks_limit: int | None,
    vocab_size: int
):
    click.echo(
        f"num_process = {num_processes}, data_file_path={data_file_path}"
    )

    start_ns = time.perf_counter_ns()
    train_bpe(
        num_processes=num_processes,
        data_file_path=data_file_path, 
        special_tokens=[END_OF_TEXT_TOKEN],
        num_chunks=num_chunks,
        num_chunks_limit=num_chunks_limit,
        vocab_size=vocab_size,
    )
    end_ns = time.perf_counter_ns()
    
    click.echo(f"Total time (in ms): {(end_ns - start_ns) / (1_000_000)}")
    
    

"""
with open(..., "rb") as f:
    num_processes = 4
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        # Run pre-tokenization on your chunk and store the counts for each pre-toke
"""

# CLI entry point
if __name__ == '__main__':
    run_train_bpe()