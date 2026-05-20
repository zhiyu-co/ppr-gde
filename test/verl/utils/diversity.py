from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
import tempfile
import gzip
import os
import lzma as xz


def compression_ratio(
        data: List[str],
        algorithm: str = 'gzip',
        verbose: bool = False,
        path: Optional[str] = None
) -> float:
    """ Calculates the compression ratio for a collection of text.
     Args:
         path (str): Path to store temporarily zipped files.
         data (List[str]): Strings to compress.
         algorithm (str, optional): Either 'gzip' or 'xz'. Defaults to 'gzip'.
         verbose (bool, optional): Print out the original and compressed size separately. Defaults to False.
     Returns:
         float: Compression ratio (original size / compressed size)
     """
     
    temp_dir = None
    if not path:
        temp_dir = tempfile.TemporaryDirectory()
        path = Path(temp_dir.name)
    else:
        path = Path(path)

    with (path / 'original.txt').open('w+') as f:
        f.write(' '.join(data))

    original_size = os.path.getsize(os.path.join(path, "original.txt"))

    if algorithm == 'gzip':

        with gzip.GzipFile(str(path / 'compressed.gz'), 'w+') as f:
            f.write(gzip.compress(' '.join(data).encode('utf-8')))

        compressed_size = os.path.getsize(os.path.join(path, "compressed.gz"))

    elif algorithm == 'xz': 

        with xz.open(str(path / 'compressed.gz'), 'wb') as f:
            f.write(' '.join(data).encode('utf-8'))

        compressed_size = (path / "compressed.gz").stat().st_size

    if verbose: 
        print(f"Original Size: {original_size}\nCompressed Size: {compressed_size}")

    if temp_dir:
        temp_dir.cleanup()

    return round(original_size / compressed_size, 3)
    
    
def generate_ngrams(tokens: List[str], n: int):
    """Generate ngrams using zip trick, no nltk."""
    return list(zip(*[tokens[i:] for i in range(n)]))


def distinct_n_scores(
        data: List[str],
        tokenizer,
        n_list=(1, 2),
):
    """
    Compute Distinct-N scores (e.g. Distinct-1, Distinct-2).

    Args:
        data (List[str]): A list of text samples.
        tokenizer (Callable): A tokenizer function that takes a string 
                              and returns a list of tokens.
                              e.g. tokenizer("hello world") -> ["hello", "world"]
        n_list (tuple): Which N values to compute. Default (1, 2).

    Returns:
        dict: { "distinct-1": float, "distinct-2": float, ... }
    """
    # ---- 1. Tokenize all data ----
    all_tokens = []
    for text in data:
        all_tokens.extend(tokenizer.tokenize(text))

    scores = {}

    # ---- 2. Calculate each distinct-n ----
    for n in n_list:
        if len(all_tokens) < n:
            scores[f"distinct-{n}"] = 0.0
            continue

        ngrams = generate_ngrams(all_tokens, n)
        unique_count = len(set(ngrams))
        total_count = len(ngrams)

        scores[f"distinct-{n}"] = round(unique_count / total_count, 3)

    return scores


def rougel_score(
        data: List[str],
        use_stemmer: Optional[str] = True,
        verbose: Optional[bool] = False,
) -> float:
    """ 
    计算rougeL 分数
    """

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=use_stemmer)

    corpus_score = 0
    
    if verbose:
        print('==> Scoring all pairs')
     
    for i, ref  in tqdm(enumerate(data), total=len(data), disable=(not verbose)):
        # Get all the other utterances to compare against a specific utterance
        preds = [x for j,x in enumerate(data) if j!=i]
        
        # Get scores over whole batch and sum it up
        doc_score = sum([scorer.score(pred, ref)['rougeL'].fmeasure for pred in preds])
        # Then average
        corpus_score += doc_score / (len(data) - 1)
    
    # case where all strings are the exact same in the list
    if corpus_score == 0: 
        corpus_score += len(data)
    
    # returns corpus level homogenization score 
    return round(corpus_score/len(data), 3)