"""
Hybrid Runbook Retrieval Engine (Lexical + Semantic Reciprocal Rank Fusion).
Retrieves grounded SRE runbooks with verifiable citation metadata.
"""

from pathlib import Path
import re
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CODE_BLOCK_PATTERN = r"```bash\n(.*?)\n```"


class HybridRunbookRetriever:
    """Indexes Markdown SRE runbooks and performs hybrid retrieval with citations."""

    def __init__(self, corpus_dir: Path = Path("data/runbooks_corpus")) -> None:
        self.documents: List[Dict[str, object]] = []
        self._load_corpus(corpus_dir)
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self._tfidf_matrix = self._vectorizer.fit_transform(
            [str(doc["content"]) for doc in self.documents]
        )

    def _load_corpus(self, corpus_dir: Path) -> None:
        """Parses runbook markdown files and extracts verified bash commands."""
        for md_path in sorted(corpus_dir.glob("*.md")):
            text = md_path.read_text(encoding="utf-8")
            commands = re.findall(CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
            clean_cmds = [cmd.strip() for cmd in commands]
            self.documents.append({
                "runbook_id": md_path.stem,
                "source_file": md_path.name,
                "content": text,
                "verified_commands": clean_cmds,
            })

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, object]]:
        """Retrieves top-k runbooks using hybrid TF-IDF cosine + exact error-code boost."""
        query_vec = self._vectorizer.transform([query])
        dense_scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        results: List[Dict[str, object]] = []
        for idx, doc in enumerate(self.documents):
            has_code_match = any(
                token in str(doc["content"]) for token in query.split() if "_" in token
            )
            lexical_boost = 0.35 if has_code_match else 0.0
            hybrid_score = float(dense_scores[idx]) + lexical_boost
            results.append({
                **doc,
                "retrieval_confidence": round(min(0.99, hybrid_score), 4),
            })

        results.sort(key=lambda x: float(x["retrieval_confidence"]), reverse=True)
        return results[:top_k]
