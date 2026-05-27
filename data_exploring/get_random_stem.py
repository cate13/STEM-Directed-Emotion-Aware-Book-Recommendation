

def _load_stem_isbns(stem_paths):
    """Load all STEM ISBNs into a set for fast lookup."""
    stem_isbns = set()

    for path in stem_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stem_isbns.add(line.strip())

    return stem_isbns

STEM_ISBNS = _load_stem_isbns(["processed_data/stem_isbns_from_classifier.txt", "processed_data/stem_isbns_from_cosine.txt", "processed_data/stem_isbns_from_topic.txt"])
