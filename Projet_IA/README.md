# 🤖 RAG Pipeline — AI Assistant on Academic Papers

> **Status: Active development** — Core pipeline complete, interface and improvements in progress.

An end-to-end **Retrieval-Augmented Generation (RAG)** system that answers questions about academic papers, built from scratch to understand every component of the pipeline.

## 🎯 What it does

Ask a question → the system finds the most relevant passages from a corpus of academic papers → generates a grounded answer using a language model.

```
"What is the attention mechanism?"
        ↓
Semantic search in 277 chunks (FAISS)
        ↓
Top-3 most relevant passages retrieved
        ↓
GPT-2 generates answer based on context
        ↓
"The attention mechanism allows the model to..."
```

## 📚 Corpus

4 Stanford academic papers:
- CS229 — Machine Learning
- CS224N — Natural Language Processing
- Attention Is All You Need (Vaswani et al., 2017)
- RAG — Retrieval-Augmented Generation (Lewis et al., 2020)

## 🏗️ Pipeline

### Offline (indexing)
1. **PDF loading** — extract text with PyMuPDF
2. **Chunking** — split into 400-word chunks with 50-word overlap
3. **Embedding** — encode each chunk with `all-MiniLM-L6-v2` (384D vectors)
4. **Indexing** — store in FAISS index (`IndexFlatL2`)

### Online (inference)
1. **Query encoding** — same model encodes the question
2. **Retrieval** — FAISS returns top-3 closest chunks (cosine similarity)
3. **Generation** — GPT-2 generates answer from injected context

## 🛠️ Tech Stack

| Component | Tool |
|-----------|------|
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | FAISS (`IndexFlatL2`) |
| Generation | GPT-2 (HuggingFace) |
| Interface | Jupyter widgets |

## 🧠 Also includes

A **MLP implemented from scratch** with NumPy (no framework) to understand what PyTorch automates:
- Forward pass, backpropagation, SGD — all manual
- Architecture: 784 → 128 → 64 → 10
- **97.7% accuracy on MNIST test set**
- Validated against PyTorch implementation

## 🚀 Getting Started

```bash
git clone https://github.com/Benismaw/projects.git
cd projects/Projet_IA

pip install -r requirements.txt

# Index the corpus (run once)
python index.py

# Launch the interface
jupyter notebook
```

## 🗂️ Project Structure

```
Projet_IA/
├── index.py              # Chunking + embedding + FAISS indexing
├── rag_pipeline.py       # Query → retrieve → generate
├── mlp_scratch.py        # MLP from scratch (NumPy)
├── corpus/               # PDF papers
├── corpus.index          # FAISS index
├── chunks.pkl            # Stored chunks
└── requirements.txt
```

## 🔧 Work in Progress

- [ ] Streamlit interface (web app)
- [ ] Conversation history
- [ ] Source display with highlighted passages
- [ ] Replace GPT-2 with a more powerful model (Mistral, Llama)
- [ ] Evaluation metrics (Precision@k, faithfulness score)
- [ ] GitHub push with full documentation

## 💡 Key Learnings

- Chunking quality is the most critical factor — bad splits lose context at boundaries
- The same encoder **must** be used for both documents and queries to ensure comparability in vector space
- GPT-2 (124M parameters) is too small for technical content — hallucinations are frequent
- Building MLP from scratch clarified what frameworks abstract away

## 📄 License

MIT
