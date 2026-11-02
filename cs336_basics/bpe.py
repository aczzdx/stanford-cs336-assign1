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


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_REGEX = re.compile(PAT)
FREQ_TABLE_PKL_FILEPATH = "data/freq_table.pkl"

END_OF_TEXT_TOKEN = "<|endoftext|>"

FrequencyTableDefualtDict = DefaultDict[bytes, int]
FrequencyTable = dict[bytes, int]

VocabTable = dict[int, bytes]

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
    data_file_path: str,
    special_tokens: list[str],
    num_chunks: int,
    num_chunks_limit: int | None = None,
    will_dump_freq_table_pkl: bool = True,
    will_load_freq_table_pkl: bool = False,
):
    freq_table: FrequencyTableDefualtDict = get_freq_table(
        num_processes, data_file_path, special_tokens, num_chunks, num_chunks_limit, will_dump_freq_table_pkl, will_load_freq_table_pkl
    )
    
    for subtoken, freq in itertools.islice(freq_table.items(), 10):
        click.echo(f"'{subtoken}' -- {freq}")

def get_freq_table(num_processes, data_file_path, special_tokens, num_chunks, num_chunks_limit, will_dump_freq_table_pkl, will_load_freq_table_pkl) -> FrequencyTableDefualtDict:
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
def run_train_bpe(
    num_processes: int,
    data_file_path: str,
    num_chunks: int,
    num_chunks_limit: int | None
):
    click.echo(
        f"num_process = {num_processes}, data_file_path={data_file_path}"
    )

    start_ns = time.perf_counter_ns()
    train_bpe(num_processes, data_file_path, [END_OF_TEXT_TOKEN], num_chunks, num_chunks_limit)
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