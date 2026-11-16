import pandas as pd
import panphon
import numpy as np


# --- Load PHOIBLE data ---
phoible = pd.read_csv("https://raw.githubusercontent.com/phoible/dev/master/data/phoible.csv", encoding="utf-8", low_memory=False)

# --- Extract phoneme inventories ---
lang1 = set(phoible[phoible.ISO6393 == 'ron']['Phoneme'].dropna())
lang2 = set(phoible[phoible.ISO6393 == 'eng']['Phoneme'].dropna())
lang3 = set(phoible[phoible.ISO6393 == 'cmn']['Phoneme'].dropna())

# --- Union of English and Chinese ---
union_lang = lang2.union(lang3)

# --- PanPhon setup ---
ft = panphon.FeatureTable()

def segment_vector(seg):
    """Return PanPhon vector or None if segment is not parseable."""
    try:
        return ft.segment_to_vector(seg)
    except Exception:
        return None

unk = set()
def feature_similarity(p1, p2):
    """Compute similarity between two phonemes based on feature vectors."""
    v1 = segment_vector(p1)
    v2 = segment_vector(p2)
    if v1 is None or v2 is None:
        if v1 is None:
            unk.add(p1)
        else:
            unk.add(p2)
        return 0.0
    # Assuming standard PanPhon vector has 24 features
    matches = sum(a == b for a, b in zip(v1, v2))
    return matches / len(v1)

def best_match_scores(setA, setB):
    scores = []
    for a in setA:
        candidates = [b for b in setB if segment_vector(b) is not None]
        if not candidates or segment_vector(a) is None:
            continue  # skip unmatched
        best = max(feature_similarity(a, b) for b in candidates)
        scores.append(best)
    return scores

# --- Compute symmetric similarity ---
print(f'Similarity Ro with En U CH: {sum(best_match_scores(lang1, union_lang)) / len(lang1)}')
print(f'Similarity Ro with En: {sum(best_match_scores(lang1, lang2)) / len(lang1)}')
print(f'Similarity Ro with Ch: {sum(best_match_scores(lang1, lang3)) / len(lang1)}')

print(f'Hard similarity: {len(lang1.intersection(union_lang)) / len(lang1)}')
print(unk)