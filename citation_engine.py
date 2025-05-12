from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EnhancedCitationGenerator:
    def __init__(self, pdf_data, num_matches=3):
        self.pdf_data = pdf_data
        self.num_matches = num_matches
        self.sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    def _chunk_sentences(self, text):
        return [s.strip() for s in text.split(".") if s.strip()]

    def generate(self, user_text):
        segments = [seg.strip() for seg in user_text.split("/") if seg.strip()]
        emb_u = self.sbert.encode(segments)
        results = {}
        for seg, u in zip(segments, emb_u):
            scores = {}
            for key, info in self.pdf_data.items():
                sents = self._chunk_sentences(info["full_text"])
                emb_s = self.sbert.encode(sents, show_progress_bar=False)
                sims = cosine_similarity([u], emb_s)[0]
                best_idx = int(np.argmax(sims))
                scores[key] = (float(sims[best_idx]), sents[best_idx])
            topk = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:self.num_matches]
            refs = [{"source": self.pdf_data[k]["nome"], "score": sc, "excerpt": ex, "page": "N/D"}
                    for k,(sc,ex) in topk]
            results[seg] = refs
        return results
