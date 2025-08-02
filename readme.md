# fully local rag chatbot for research

ask questions about your papers locally. no apis, no data sent anywhere.

## features

- **100% local**: No api keys or internet required (except initial downloads)
- **simple setup**: No complex configuration
- **flexible models**: ollama (gpu) and sentenceTransformers (cpu)
- **research-focused**: Optimized for scientific papers and technical documents

## quick start

### 1. clone and setup
```bash
git clone https://github.com/rizanB/lusc_rag.git
cd lusc_rag
```

### 2. install dependencies
```bash
pip install streamlit langchain-community langchain-core sentence-transformers chromadb
```

### 3. choose model

**cpu-only (SentenceTransformers): fast but basic, text-only searches**
```bash
streamlit run main.py
```

**gpu-accelerated (ollama - recommended): slightly slower, but better responses**
```bash
# install ollama
curl -fsSL https://ollama.ai/install.sh | sh

# download models (~2gb total)
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# run app
streamlit run main.py
```

### 4. add your papers
1. create a folder called `recent_advances/` in project directory
2. add your research papers as `.txt` files (large files are chunked automatically)
3. click "load/reload Documents" (takes 3-5min for 12 papers)

##performance
**first time: ~5min to process 12 papers**
**after that: instant startup, 10-15 sec responses**
**gpu usage: ~3000mb with llama3.2:3b**

## usage

1. **load documents**: click the button to process your text files
2. **ask questions**: ask the chatbot about your papers and it answers with references

### example questions
- "summarize the machine learning approaches mentioned"
- "what are the key findings about personalized medicine?"
- "what clustering techniques are in use in lung cancer?"

## project structure

```
.
├── chroma_store
│   └── chroma.sqlite3
├── chroma_store_local
│   ├── 15a05497-92de-4b30-8d74-8beaf9376b08
│   └── chroma.sqlite3
├── chroma_store_ollama
│   ├── 566c849c-2cd7-4da8-904c-89dc5c347b8f
│   └── chroma.sqlite3
├── conversation_log.txt #history of past conversations
├── main.py #code here
├── readme.md
└── recent_advances #folder with papers, add new papers here
    ├── chakraborty-2018-onco-multi-omics.txt
    ├── chehelgerdi-2024-cripsr-cancer.txt
    ├── chen-2024-automatic-lung-cancer-subtyping-with-rapid-on-site-evaluation-slides-and-serum-biological-markers.txt
    ├── kori-2024-biomarkers-in-silico-pre-clinical-validation.txt
    ├── leader-2021-sc-nsclc.txt
    ├── li-2022-profiling-nsclc-by-single-cell-rna-seq.txt
    ├── li-2024-non-invasive-subtype-tumor-signatures-cfDNA-methylome.txt
    ├── ma-2025-liquid-biopsy-vesicle-protein-biomarkers-for-lusc.txt
    ├── pan-2024-blood-biomarkers-for-efficacy-of-neoadjuvant-chemo-immuno-therapy-nsclc.txt
    ├── patel-2020-ai-decode-cancer.txt
    ├── torok-2025-serum-and-exosome-wnt5a-as-biomarkers-in-nsclc.txt
    └── zang-2025-label-free-subtyping-deep-classification-virtual-ihc-staining.txt
```

## system requirements

**minimum (CPU only):**
- 4GB RAM
- Python 3.8+

**recommended (GPU accelerated):**
- 16GB RAM
- 4GB+ GPU VRAM, 1650 card or higher

## common errors

**Ollama not found:**
- did you install ollama? try 'ollama list' to see available models

**Slow responses:**
- this chatbot uses llama3.2:3b by default. Use the 1b model or reduce document chunks in the search.

## technologies

- **streamlit**: interface
- **langchain**: RAG framework
- **ollama**: for local models
- **sentenceTransformers**: embeddings for cpu fallback
- **chromaDB**: vector database

## license

MIT, free to use. 
Let me know if you have any feature requests.

---

*for researchers who want to chat about their papers privately and locally*