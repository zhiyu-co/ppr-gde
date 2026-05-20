from itertools import chain
import sacrebleu


def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]
        

def distinct_n_sentence_level(sentence, n):
    """ 句子级别 distinct-n, 越大重复度越小 """
    length = len(sentence)
    if length < n:
        return 0.0
    ngrams_list = list(ngrams(sentence, n))
    return len(set(ngrams_list)) / len(ngrams_list)


def distinct_n_corpus_level(sentences, n):
    """
    句子间 distinct-n, 返回值越低，丰富度越小, 推荐n=1和n=2
    args:
        sentences: token形式的句子集合
    return:
        score: 最终 distinct-n 分数
    """
    all_ngrams = []

    for sentence in sentences:
        if len(sentence) >= n:
            all_ngrams.extend(list(ngrams(sentence, n)))

    if len(all_ngrams) == 0:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def self_bleu(sentences, max_n=4, uid=0):
    """
    返回值越高，丰富度越低
    计算 Self-BLEU: 对每个句子，把其余句子作为参考，计算 BLEU，然后求平均。
    sentences: List[str]，生成的句子集合, 要求分词后版本
    max_n: 最高 n-gram，比如 4 表示计算 BLEU-1 到 BLEU-4
    返回: dict, keys 类似 "BLEU-1", "BLEU-2", ..., "Self-BLEU-1", ...
    """
    joined = ["".join(s) for s in sentences]
    joined = [s for s in joined if len(s.strip()) > 0]
    if len(joined) < 2:
        print(f"BLEU错误: uid({uid}) 不包含足够的response样本, 长度为 {len(joined)}")
        return {"Self-BLEU-1": 0, "Self-BLEU-2": 0, "Self-BLEU-3": 0, "Self-BLEU-4": 0}
    results = {}
    # sacrebleu.CorpusBLEU 期望 list of hypotheses 和 list of list of references
    # 但我们这里要一个一个句子做 hypothesis
    bleu_scores = {n: [] for n in range(1, max_n+1)}

    for i, hyp in enumerate(joined): 
        refs = [joined[j] for j in range(len(joined)) if j != i]
        ref_lists = [[r] for r in refs]
        
        # 计算 BLEU
        bleu = sacrebleu.corpus_bleu([hyp], ref_lists, tokenize='none', smooth_method="exp")
        
        precisions = bleu.precisions
        for n in range(1, max_n + 1):
            bleu_scores[n].append(precisions[n-1])

    # 计算平均
    for n in range(1, max_n + 1):
        results[f"Self-BLEU-{n}"] = sum(bleu_scores[n]) / len(bleu_scores[n])

    return results
