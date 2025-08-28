# LightRAG-reimplementation
A simplified reimplementation of LightRAG as an independent study.

## Description
This repository documents my simplified reimplementation of **LightRAG** presented in the paper **"[Lightrag: Simple and fast retrieval-augmented generation](https://arxiv.org/abs/2410.05779)" by Guo et al.(2024)** during **August 2025**.
- LLM-based entities and relations extraction for **Graph-based Text Indexing**
- NetworkX and FAISS for **Dual-level Retrieval**
- Comprehensive context for **Retrieval-augmented Answer Generation**

For more implementation details and experimental results, please refer to the [reimplementation report](./report/report.pdf).

## Usage
Clone the repository and set up the environment using Conda:
```bash
conda env create -f environment.yml
```
Then create a `.env` file in the working directory to set up your API key.
### To try out the model:
- Configure `simplified_lightrag.ipynb` to use your LLM
- Modify `filename` to load a `.txt` file in the working directory
- Run the Jupyter Notebook to start a conversation session
### To replicate the experiments:
- Configure `simplified_lightrag.py`, `naive_rag.py` and `evaluation.py` to use your LLM
- Download `mix.jsonl` from the [UltraDomain Benchmark](https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/mix.jsonl?download=true) into the working directory
- Modify `evaluation.py` to select which models to compare
- Run `evaluation.py` to start the evaluation
