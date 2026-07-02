"""Standalone text generation metrics used by graph experiments."""

from __future__ import annotations

import collections
import itertools
import math
import re
from collections import Counter

import numpy as np


DEFAULT_STOP_TOKENS = {
    "", ".", ",", "!", "?", ":", ";", "(", ")", "'s", "'m", "'ve",
    "n't", "'re", "'d", "'ll",
    "a", "an", "and", "are", "as", "at", "be", "but", "by",
    "for", "from", "he", "her", "his", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "our", "she", "that",
    "the", "their", "them", "there", "they", "this", "to",
    "was", "we", "were", "with", "you", "your",
}


def _get_ngrams(segment, max_order):
    counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            counts[tuple(segment[i:i + order])] += 1
    return counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for references, translation in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)
        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        overlap = _get_ngrams(translation, max_order) & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        elif possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]

    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length if reference_length else 0.0
    if ratio > 1.0:
        bp = 1.0
    elif ratio == 0:
        bp = 0
    else:
        bp = math.exp(1 - 1.0 / ratio)
    return geo_mean * bp, precisions, bp, ratio, translation_length, reference_length


def bleu_score(references, generated, n_gram=4, smooth=False):
    formatted_ref = [[ref] for ref in references]
    bleu_s, _, _, _, _, _ = compute_bleu(formatted_ref, generated, n_gram, smooth)
    return bleu_s * 100


def _split_into_words(sentences):
    return list(itertools.chain(*[sentence.split(" ") for sentence in sentences]))


def _rouge_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    for i in range(text_length - n + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _word_ngrams(n, sentences):
    assert len(sentences) > 0
    assert n > 0
    return _rouge_ngrams(n, _split_into_words(sentences))


def rouge_n(evaluated_sentences, reference_sentences, n=2):
    evaluated_ngrams = _word_ngrams(n, evaluated_sentences)
    reference_ngrams = _word_ngrams(n, reference_sentences)
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)
    overlapping_count = len(evaluated_ngrams.intersection(reference_ngrams))
    precision = 0.0 if evaluated_count == 0 else overlapping_count / evaluated_count
    recall = 0.0 if reference_count == 0 else overlapping_count / reference_count
    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return f1_score, precision, recall


def _lcs_table(x, y):
    table = {}
    for i in range(len(x) + 1):
        for j in range(len(y) + 1):
            if i == 0 or j == 0:
                table[i, j] = 0
            elif x[i - 1] == y[j - 1]:
                table[i, j] = table[i - 1, j - 1] + 1
            else:
                table[i, j] = max(table[i - 1, j], table[i, j - 1])
    return table


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
    reference_words = _split_into_words(reference_sentences)
    evaluated_words = _split_into_words(evaluated_sentences)
    m = len(reference_words)
    n = len(evaluated_words)
    if m == 0 or n == 0:
        return 0.0, 0.0, 0.0
    llcs = _lcs_table(evaluated_words, reference_words)[n, m]
    r_lcs = llcs / m
    p_lcs = llcs / n
    beta = p_lcs / (r_lcs + 1e-12)
    num = (1 + beta ** 2) * r_lcs * p_lcs
    denom = r_lcs + (beta ** 2) * p_lcs
    f_lcs = num / (denom + 1e-12)
    return f_lcs, p_lcs, r_lcs


def rouge(hypotheses, references):
    rouge_1 = [rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)]
    rouge_2 = [rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)]
    rouge_l = [
        rouge_l_sentence_level([hyp], [ref])
        for hyp, ref in zip(hypotheses, references)
    ]
    rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))
    rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))
    rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))
    return {
        "rouge_1": rouge_1_r,
        "rouge_2": rouge_2_r,
        "rouge_l": rouge_l_f,
    }


def rouge_score(references, generated):
    return {key: value * 100 for key, value in rouge(generated, references).items()}


def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        if not any(two_seq_same(seq, uni_seq) for uni_seq in unique_seq):
            unique_seq.append(seq)
    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def feature_detect(seq_batch, feature_set):
    feature_batch = []
    for ids in seq_batch:
        feature_batch.append({token for token in ids if token in feature_set})
    return feature_batch


def feature_matching_ratio(feature_batch, test_feature):
    count = 0
    for fea_set, fea in zip(feature_batch, test_feature):
        if fea in fea_set:
            count += 1
    return count / len(feature_batch)


def feature_coverage_ratio(feature_batch, feature_set):
    features = set()
    for fb in feature_batch:
        features = features | fb
    return len(features) / len(feature_set)


def feature_diversity(feature_batch):
    list_len = len(feature_batch)
    total_count = 0
    for i, x in enumerate(feature_batch):
        for y in feature_batch[i + 1:]:
            total_count += len(x & y)
    denominator = list_len * (list_len - 1) / 2
    return total_count / denominator


def postprocessing(string):
    string = re.sub("'s", " 's", string)
    string = re.sub("'m", " 'm", string)
    string = re.sub("'ve", " 've", string)
    string = re.sub("n't", " n't", string)
    string = re.sub("'re", " 're", string)
    string = re.sub("'d", " 'd", string)
    string = re.sub("'ll", " 'll", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(",+", " , ", string)
    string = re.sub(":+", " , ", string)
    string = re.sub(";+", " . ", string)
    string = re.sub(r"\.+", " . ", string)
    string = re.sub("!+", " ! ", string)
    string = re.sub(r"\?+", " ? ", string)
    string = re.sub(" +", " ", string).strip()
    return string


def ids_clear(ids, pad_token_id=0, eos_token_ids=(2,), skip_token_ids=(1,)):
    eos_token_ids = set(eos_token_ids or [])
    skip_token_ids = set(skip_token_ids or [])
    tokens = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id in skip_token_ids:
            continue
        if token_id == pad_token_id or token_id in eos_token_ids:
            break
        tokens.append(token_id)
    return tokens


def ids2words(ids, tokenizer):
    text = tokenizer.decode(ids)
    return [token for token in postprocessing(text).split()]


def compute_tail_demand_from_tokens(token_batches, stop_tokens=None):
    if stop_tokens is None:
        stop_tokens = DEFAULT_STOP_TOKENS
    filtered_batches = [
        [token for token in tokens if token and token not in stop_tokens]
        for tokens in token_batches
    ]
    token_counts = Counter()
    for tokens in filtered_batches:
        token_counts.update(tokens)
    sorted_tokens = [token for token, _ in token_counts.most_common()]
    if len(sorted_tokens) == 0:
        return [0.0 for _ in filtered_batches]
    base_size, remainder = divmod(len(sorted_tokens), 5)
    group_sizes = [base_size + (1 if idx < remainder else 0) for idx in range(5)]
    head_tokens = set(sorted_tokens[:group_sizes[0]])
    demands = []
    for tokens in filtered_batches:
        if len(tokens) == 0:
            demands.append(0.0)
            continue
        tail_count = sum(1 for token in tokens if token not in head_tokens)
        demands.append(tail_count / len(tokens))
    return demands


def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0
    unique = set()
    for i in range(len(sentence) - n + 1):
        unique.add(tuple(sentence[i:i + n]))
    return len(unique) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    if not sentences:
        return 0.0
    return sum(distinct_n_sentence_level(s, n) for s in sentences) / len(sentences)


def ngram_entropy(sentences, n):
    counts = Counter()
    total = 0
    for s in sentences:
        for i in range(len(s) - n + 1):
            counts[tuple(s[i:i + n])] += 1
            total += 1
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        h -= p * math.log2(p)
    return h


def corpus_diversity(sentences):
    """sentences: list[list[str]] (token lists of generated text).
    Returns (Distinct-1, Distinct-2, ENTR) matching CIER/MAPLE."""
    d1 = distinct_n_corpus_level(sentences, 1)
    d2 = distinct_n_corpus_level(sentences, 2)
    entr = (
        ngram_entropy(sentences, 1)
        + ngram_entropy(sentences, 2)
        + ngram_entropy(sentences, 3)
    ) / 3.0
    return d1, d2, entr


def assign_tail_demand_groups(demands, low_frac=0.20, high_frac=0.20):
    if low_frac < 0 or high_frac < 0 or low_frac + high_frac > 1:
        raise ValueError("low_frac and high_frac must be non-negative and sum to at most 1")

    n = len(demands)
    low_count = int(math.floor(n * low_frac))
    high_count = int(math.floor(n * high_frac))
    ranked = sorted(range(n), key=lambda idx: (demands[idx], idx))
    labels = ["excluded"] * n
    for idx in ranked[:low_count]:
        labels[idx] = "low"
    for idx in ranked[n - high_count:]:
        labels[idx] = "high"
    return labels
