import os
import sys
import bibtexparser
import pandas as pd
from huggingface_hub import snapshot_download

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

from transformers import pipeline
from llama_cpp import Llama

# ── CONFIGURAÇÃO DO MODELO REMOTO ───────────────────────────
HF_MODEL_ID = "marcosgoncaf/mistral-7b-instruct-v0.2-Q4_K_M"
NUM_THREADS = max(1, os.cpu_count() // 2)
_cached_gguf = None

def get_local_gguf():
    global _cached_gguf
    if _cached_gguf is None:
        repo_dir = snapshot_download(repo_id=HF_MODEL_ID, repo_type="model")
        # encontra o .gguf
        for fname in os.listdir(repo_dir):
            if fname.endswith(".gguf"):
                _cached_gguf = os.path.join(repo_dir, fname)
                break
        if _cached_gguf is None:
            sys.exit("Erro: não achei arquivo .gguf no snapshot")
    return _cached_gguf

def load_all_metadata(files):
    records = []
    for f in files:
        name, data = f.name, f.read()
        if name.lower().endswith(".csv"):
            df = pd.read_csv(pd.io.common.BytesIO(data))
            df["source_file"] = name
            records += df.to_dict(orient="records")
        else:
            text = data.decode("utf-8", errors="ignore")
            bib = bibtexparser.loads(text)
            for entry in bib.entries:
                entry["source_file"] = name
                records.append(entry)
    df = pd.DataFrame(records)
    if "title" not in df.columns:
        df["title"] = df.apply(
            lambda r: next((str(r[c]) for c in df.columns if isinstance(r[c], str) and len(r[c])>50),
                           "Sem título"),
            axis=1
        )
    if "abstract" not in df.columns:
        text_cols = [c for c in df.columns if df[c].dtype == object]
        best = max(text_cols, key=lambda c: df[c].astype(str).map(len).mean())
        df["abstract"] = df[best].astype(str)
    df = df[df["abstract"].astype(bool)].reset_index(drop=True)
    if "doi" in df.columns:
        df = df.drop_duplicates(subset="doi")
    else:
        df = df.drop_duplicates(subset="title")
    return df[["title","abstract","source_file"]]

def summarize_extractive(texts, n_sentences=3):
    tr = TextRankSummarizer()
    out = []
    for txt in texts:
        parser = PlaintextParser.from_string(txt, Tokenizer("portuguese"))
        sents = tr(parser.document, n_sentences)
        out.append(" ".join(str(s) for s in sents))
    return out

# ── FLAN-T5-SMALL ────────────────────────────────────────────
_hf_pipe = None
def summarize_flant5(texts, max_len=150):
    global _hf_pipe
    if _hf_pipe is None:
        _hf_pipe = pipeline("summarization", model="google/flan-t5-small", device=-1)
    return [_hf_pipe(txt, max_length=max_len, truncation=True)[0]["summary_text"] for txt in texts]

# ── MISTRAL QUANTIZADO ───────────────────────────────────────
_llama = None
def summarize_llama(texts, max_len=150):
    global _llama
    if _llama is None:
        gguf = get_local_gguf()
        _llama = Llama(model_path=gguf, n_threads=NUM_THREADS, n_ctx=256)
    out = []
    for txt in texts:
        prompt = f"Resuma em até {max_len} tokens:\n\n{txt}"
        r = _llama(prompt, max_tokens=max_len)
        out.append(r["choices"][0]["text"].strip())
    return out
