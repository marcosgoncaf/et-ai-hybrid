import streamlit as st
import fitz, hashlib
from summarizer import load_all_metadata, summarize_extractive, summarize_flant5, summarize_llama
from citation_engine import EnhancedCitationGenerator

st.set_page_config(page_title="et.AI Hybrid", page_icon="🧠", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
  background-image: url('logo.png');
  background-size: 200px;
  background-position: top right;
  background-color: rgba(255, 250, 245, 0.9);
}
.stButton>button { background-color: #BFD8B8; color: #3D405B; }
</style>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔖 Resumo", "📑 Citações"])

with tabs[0]:
    st.header("Geração de Resumos")
    uploaded = st.file_uploader("Carregue CSV ou BibTeX", type=["csv","bib"], accept_multiple_files=True)
    if uploaded:
        df = load_all_metadata(uploaded)
        texts = df["abstract"].tolist()
        mode = st.selectbox("Modo de resumo:",
            ["Extractivo (ultra-rápido)", "Abstrativo leve (Flan-T5-small)", "Abstrativo avançado (Mistral Q4_K_M)"])
        if st.button("Gerar Resumo"):
            with st.spinner("Resumindo…"):
                if mode.startswith("Extractivo"):
                    out = summarize_extractive(texts)
                elif "Flan" in mode:
                    out = summarize_flant5(texts)
                else:
                    out = summarize_llama(texts)
            st.markdown("### Resumo Consolidado")
            st.write("\n\n".join(out))

with tabs[1]:
    st.header("Gerador de Citações")
    pdfs = st.file_uploader("Carregue PDFs", type="pdf", accept_multiple_files=True)
    if pdfs:
        if 'pdf_data' not in st.session_state:
            st.session_state.pdf_data = {}
        for f in pdfs:
            data = f.read(); key = hashlib.md5(data).hexdigest()
            if key not in st.session_state.pdf_data:
                doc = fitz.open(stream=data, filetype="pdf")
                st.session_state.pdf_data[key] = {
                    "nome": f.name,
                    "full_text": "\n".join(p.get_text("text") for p in doc)
                }
        st.success(f"{len(st.session_state.pdf_data)} PDFs carregados!")
        usertxt = st.text_area("Cole o texto (use '/' para separar):", height=150)
        if st.button("Gerar Citações"):
            gen = EnhancedCitationGenerator(st.session_state.pdf_data)
            out = gen.generate(usertxt)
            for seg, refs in out.items():
                st.subheader(f"Trecho: {seg}")
                for r in refs:
                    st.markdown(f"**Fonte:** {r['source']}  \n"
                                f"**Score:** {r['score']:.2f}  \n"
                                f"> {r['excerpt']}")
