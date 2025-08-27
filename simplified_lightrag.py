# Simplified LightRAG
# This implementation provides a minimal working version of the LightRAG architecture as described in the original paper: 
# "LightRAG: Simple and Fast Retrieval-Augmented Generation" by Guo et al. https://arxiv.org/abs/2410.05779
# 
# This Python version process queries in batch to facilitate evaluation on a given dataset.
# Another version (./simlified_lightrag.ipynb) receives user queries sequentially.

from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import TokenTextSplitter

from collections import defaultdict

import networkx as nx
import regex as re
import pickle
import uuid

from prompt import *
import rag
import utils

import warnings
warnings.filterwarnings("ignore", message="Relevance scores must be between 0 and 1")


def simplified_lightrag(
        input: str, 
        context: str,
        query_mode: str
    ):
    load_dotenv(".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

    # Embedding function initialization
    embedding_function = OllamaEmbeddings(
        model="bge-m3:567m",
    )

    text_splitter = TokenTextSplitter(chunk_size=1200)
    chunks = text_splitter.split_text(context)

    # Extraction
    tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"] 
    record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]  
    completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"] 

    prompts = []
    for chunk in chunks:
        prompt_template = PROMPTS["entity_extraction"]
        examples_template = PROMPTS["entity_extraction_examples"]
        examples = "\n".join(examples_template).format(
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
        )
        prompt = prompt_template.format(
            language="English",
            entity_types=PROMPTS["DEFAULT_ENTITY_TYPES"],
            tuple_delimiter=tuple_delimiter,
            record_delimiter=record_delimiter,
            completion_delimiter=completion_delimiter,
            examples=examples,
            input_text=chunk
        )
        prompts.append(prompt)
    responses = llm.batch(prompts)

    # Parsing extraction output
    graph_nodes = defaultdict(lambda: {"descriptions": [], "type": "", "chunk_ids": []})
    graph_edges = defaultdict(lambda: {"descriptions": [], "keywords": [], "weight": 1.0, "chunk_ids": []})
    for chunk_id, response in enumerate(responses):
        response = response.content
        
        records = utils.split_string_by_multi_markers(response, [record_delimiter, completion_delimiter])

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue
            record = match.group(1)
            record_attributes = utils.split_string_by_multi_markers(record, tuple_delimiter)

            # Parse entity
            entity_data = utils._handle_single_entity_extraction(record_attributes, llm)
            if entity_data:
                name = entity_data["entity_name"]
                type = entity_data["entity_type"]
                description = entity_data["description"]

                # Deduplication by name
                graph_nodes[name]["descriptions"].append(description)
                graph_nodes[name]["chunk_ids"].append(chunk_id)

            # Parse relation    
            relation_data = utils._handle_single_relation_extraction(record_attributes, llm)
            if relation_data:
                source = relation_data["src_id"]
                target = relation_data["tgt_id"]
                weight = relation_data["weight"]
                description = relation_data["description"]
                keywords = relation_data["keywords"]

                key = (source, target)
                graph_edges[key]["keywords"].append(keywords)
                graph_edges[key]["weight"] = weight

                # Deduplication by key
                graph_edges[key]["descriptions"].append(description)
                graph_edges[key]["chunk_ids"].append(chunk_id)


    # Deduplication
    temp_entities_vectorstore = Chroma(
            collection_name="temp_entities",
            embedding_function=embedding_function,
            persist_directory="./graph",
            collection_metadata={"hnsw:space": "cosine"}
        )
    
    # Clear the temporary collection
    temp_entities_vectorstore.delete_collection()
    
    temp_entities_vectorstore = Chroma(
        collection_name="temp_entities",
        embedding_function=embedding_function,
        persist_directory="./graph",
        collection_metadata={"hnsw:space": "cosine"}
    )
        
    temp_entities_vectorstore.add_texts([str(key) for key in graph_nodes.keys()])

    similarity_threshold = 0.7 # tunable

    # Cluster entities by names
    entity_clusters = {}
    processed_names = set()
    for name in graph_nodes.keys():
        if name in processed_names:
            continue

        search_results = temp_entities_vectorstore.similarity_search_with_relevance_scores(name, k=5)
        
        cluster = [name]
        processed_names.add(name)
        for document, score in search_results:
            variant = document.page_content
            if score > similarity_threshold and name != variant:
                cluster.append(variant)
                processed_names.add(variant)
        entity_clusters[name] = sorted(list(set(cluster)))
    
    name_to_canonical = {}
    for name, variants in entity_clusters.items():
        canonical = name
        for variant in variants:
            name_to_canonical[variant] = canonical
    

    merged_graph_nodes = defaultdict(lambda: {"canonical": "", "type": "", "descriptions": [], "chunk_ids": []})
    merged_graph_edges = defaultdict(lambda: {"descriptions": [], "keywords": [], "weight": 1.0, "count": 0, "chunk_ids": []})

    # Merge entities
    for name, variants in entity_clusters.items():
        descriptions = []
        for variant in variants:
            if variant in graph_nodes:
                descriptions.extend(graph_nodes[variant]["descriptions"])

        new_key = tuple(variants)
        merged_graph_nodes[new_key] = {
            "canonical": name,
            "type": graph_nodes[name]["type"],
            "descriptions": descriptions,
            "chunk_ids": graph_nodes[name]["chunk_ids"]
        }
    
    # Reconnect relations
    for key, data in graph_edges.items():
        source, target = key
        canonical_source = name_to_canonical.get(source)
        canonical_target = name_to_canonical.get(target)

        if canonical_source and canonical_target:
            new_key = (canonical_source, canonical_target)
            merged_graph_edges[new_key]["descriptions"].extend(data["descriptions"]) 
            merged_graph_edges[new_key]["keywords"].extend(data["keywords"])
            merged_graph_edges[new_key]["weight"] += data["weight"] 
            merged_graph_edges[new_key]["count"] += 1 
            merged_graph_edges[new_key]["chunk_ids"].extend(data["chunk_ids"])
    
    for key in merged_graph_edges:
        if merged_graph_edges[new_key]["count"] > 1: 
            merged_graph_edges[new_key]["weight"] /= merged_graph_edges[new_key]["count"] 

    graph_nodes = merged_graph_nodes
    graph_edges = merged_graph_edges
    
    # Summarization
    entity_or_relation_names = []
    entity_or_relation_descriptions = []

    for name, data in graph_nodes.items():
        entity_or_relation_names.append(", ".join(name))
        entity_or_relation_descriptions.append(" ".join(data["descriptions"]))
    for key, data in graph_edges.items():
        entity_or_relation_names.append(" -> ".join(key))
        entity_or_relation_descriptions.append(" ".join(data["descriptions"]))

    summaries = utils._handle_entity_relation_summary(entity_or_relation_names, entity_or_relation_descriptions, "", llm)


    G = nx.DiGraph()
    nodes = []
    edges = []
    summary_index = 0

    for name_tuple, data in graph_nodes.items():
        summary = summaries[summary_index].content
        canonical = data["canonical"]

        # Add node to graph
        G.add_node(
                canonical,
                graph_id=canonical, 
                type=data['type'], 
                description=summary,
                chunk_ids=data["chunk_ids"]
        )

        nodes.append(
            Document(
                page_content=summary, 
                metadata={
                    "graph_id": canonical,
                    "type": data["type"],
                    "value": summary,
                    "chunk_ids": " ".join([str(id) for id in data["chunk_ids"]])
                }
            )
        )
    
        summary_index += 1

    for key, data in graph_edges.items():
        summary = summaries[summary_index].content
        source, target = key

        edge_id = str(uuid.uuid4())

        G.add_edge(
                source, 
                target, 
                graph_id=edge_id, 
                description=summary, 
                keywords=data['keywords'], 
                weight=float(data['weight']),
                chunk_ids=data["chunk_ids"]
        )
        
        edges.append(
            Document(
                page_content=summary,
                metadata={
                    "graph_id": edge_id, 
                    "value": summary,
                    "source": source, 
                    "target": target,
                    "weight": data["weight"],
                    "chunk_ids": " ".join([str(id) for id in data["chunk_ids"]])
                }
            )
        )
        summary_index += 1

    # Populate vectorstore
    entities_vectorstore = FAISS.from_documents(documents=nodes, embedding=embedding_function)
    relations_vectorstore = FAISS.from_documents(documents=edges, embedding=embedding_function)


    # Query keywords extraction
    extracted_items = rag.extract_query([], llm, query_mode, input)
    query, query_mode, hl_keywords, ll_keywords = extracted_items

    # Dual level retrieval
    retrieved_entities_and_relations = rag.dual_level_retrieval(
        G,
        entities_vectorstore,
        relations_vectorstore,
        hl_keywords,
        ll_keywords,
        query_mode
    )

    context_str = rag.build_context(retrieved_entities_and_relations, [Document(chunk) for chunk in chunks])

    history = {"role": "user", "content": query}
    # Final query
    prompt_template = PROMPTS["rag_response"]
    final_query = prompt_template.format(
        history=history,
        context_data=context_str,
        response_type="",
        user_prompt=""
    )

    # Final generation
    answer = llm.invoke(final_query)

    return answer


