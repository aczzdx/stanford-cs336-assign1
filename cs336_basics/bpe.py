import multiprocessing as mp
import os
import time
import re
from typing import Any, BinaryIO
# import tqdm based on different platform
from tqdm.std import trange
import click


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
END_OF_TEXT_TOKEN = b"<|endoftext|>"

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
            
            if first_found_at:
                chunk_boundaries[bi] = initial_position + first_found_at
                break
            initial_position += mini_chunk_size
            

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize(idx: int, start: int, end: int, data_file_path: str, q: mp.Queue) -> None:
    
    with open(data_file_path, "rb") as f:
            
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    test_msg = f"chunk_size={len(chunk)}"
    ret: tuple[int, list[bytes]] = (idx, [test_msg.encode('utf-8')])
    q.put(ret)

def init_vocab() -> dict[int, bytes]:
    ret: dict[int, bytes] = {}
    
    # EndOfText
    ret[0] = END_OF_TEXT_TOKEN
    
    for b in range(1 << 8):
        ret[b + 1] = chr(b).encode('utf-8')
    
    return ret

def queue_listener_merger(q: mp.Queue) -> dict[int, bytes]:
    
    ret = init_vocab()
    
    return ret
    
def train_bpe(
    num_processes: int,
    data_file_path: str,
    num_chunks: int
):
    with open(data_file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, [END_OF_TEXT_TOKEN])

    with mp.Pool(num_processes) as pool, mp.Manager() as manager:
        q = manager.Queue()
        params = [
            (i, start_end_pair[0], start_end_pair[1], data_file_path, q)
            for i, start_end_pair in enumerate(zip(boundaries[:-1], boundaries[1:]))
        ]
        
        pool.starmap(pretokenize, params)
        
        
@click.command()
@click.option('--num-processes', default=4)
@click.option('--data-file', 'data_file_path', type=click.Path(exists=True))
@click.option('--num-chunks', default=20_000)
def run_train_bpe(
    num_processes: int,
    data_file_path: str,
    num_chunks: int,
):
    click.echo(
        f"num_process = {num_processes}, data_file_path={data_file_path}"
    )

    start_ns = time.perf_counter_ns()
    train_bpe(num_processes, data_file_path, num_chunks)
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