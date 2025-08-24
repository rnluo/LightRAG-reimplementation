from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from collections import defaultdict
import networkx as nx
from langchain_community.vectorstores import FAISS
from prompt import *
import json
import json_repair

def extract_query(
    history: list,
    llm: ChatOpenAI
):  
    print("Enter query mode (local/global).\nAny other input defaults to hybrid mode.")
    query_mode = input("Mode: ")

    if not query_mode == "local" or query_mode == "global":
        query_mode = "hybrid"

    print("Enter your query.\nEntering nothing defaults to a preset query.")
    query = input("Query: ")

    if query == "":
        query = "Compare and contrast the different theories about love mentioned in the text, illustrating how each of them helps conveying the main philosophical thought."
    
        
    prompt_template = PROMPTS["keywords_extraction"]
    examples = PROMPTS["keywords_extraction_examples"]
    prompt = prompt_template.format(
        examples=examples,
        history=build_history_context(history),
        query=query
    )

    response = llm.invoke(prompt)

    try:
        keywords_data = json_repair.loads(response.content)
        if not keywords_data:
            print("No JSON-like structure found in the LLM response.")
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"LLM response: {keywords_data}")

    hl_keywords = keywords_data.get("high_level_keywords", [])
    ll_keywords = keywords_data.get("low_level_keywords", [])

    return query, query_mode, hl_keywords, ll_keywords

def menu():
    print("Press 'ENTER' to continue.\nPress 'q' to quit.")
    operation = input("Input: ")
    if operation == 'q':
        return False
    return True

def manage_history(
    history: list,
    role: str,
    content: str
):
    entry = {"role": role, "content": content}
    history.append(entry)

def build_history_context(
    history: list,
):
    return "\n".join([f"{entry['role']}: {entry['content']}" for entry in history])
    
def dual_level_retrieval(
        G: nx.DiGraph,
        entities_vectorstore: FAISS,
        relations_vectorstore: FAISS,
        hl_keywords: list[str],
        ll_keywords: list[str],
        query_mode: str,
):  
    local_entities = []
    local_relations = []
    global_entities = []
    global_relations = []

    if query_mode == "local":
        for ll_keyword in ll_keywords:
            local_entities.extend(entities_vectorstore.similarity_search(ll_keyword, k=3))
            graph_ids = [entity.metadata.get("graph_id") for entity in local_entities if entity.metadata.get("graph_id") is not None]
            
            for graph_id in graph_ids:
                for neighbor in G.neighbors(graph_id):
                    edge_data = G.get_edge_data(graph_id, neighbor)
                    if edge_data:
                        local_relations.append(edge_data)

    elif query_mode == "global":
        for hl_keyword in hl_keywords:
            global_relations.extend(relations_vectorstore.similarity_search(hl_keyword, k=3))
            graph_ids = [relation.metadata.get("graph_id") for relation in global_relations]

            for relation in global_relations:
                source = relation.metadata.get("source")
                target = relation.metadata.get("target")
                if source and source in G.nodes:
                    global_entities.append(G.nodes[source])
                if target and target in G.nodes:
                    global_entities.append(G.nodes[target])

    else:
        for ll_keyword in ll_keywords:
            local_entities.extend(entities_vectorstore.similarity_search(ll_keyword, k=3))
            graph_ids = [entity.metadata.get("graph_id") for entity in local_entities if entity.metadata.get("graph_id") is not None]
            
            for graph_id in graph_ids:
                for neighbor in G.neighbors(graph_id):
                    edge_data = G.get_edge_data(graph_id, neighbor)
                    if edge_data:
                        local_relations.append(edge_data)
        
        for hl_keyword in hl_keywords:
            global_relations.extend(relations_vectorstore.similarity_search(hl_keyword, k=3))
            graph_ids = [relation.metadata.get("graph_id") for relation in global_relations]

            for relation in global_relations:
                source = relation.metadata.get("source")
                target = relation.metadata.get("target")
                if source and source in G.nodes:
                    global_entities.append(G.nodes[source])
                if target and target in G.nodes:
                    global_entities.append(G.nodes[target])

    # Make all four lists type <class 'dict'>
    local_entities = [relation.metadata for relation in local_entities]
    global_relations = [entity.metadata for entity in global_relations]

    return [local_entities, local_relations, global_entities, global_relations]

# adapted from https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py
def build_context(
    retrieved_entities_and_relations: list[list[dict]],
    chunks: list[Document]
):
    local_entities, local_relations, global_entities, global_relations = retrieved_entities_and_relations

    chunk_ids = set()

    # Round-robin merge entities
    final_entities = []
    seen_ids = set()

    max_len = max(len(local_entities), len(global_entities))
    for i in range(max_len):
        # First from local
        if i < len(local_entities):
            entity = local_entities[i]
            entity_id = entity.get("graph_id")
            if entity_id and entity_id not in seen_ids:
                final_entities.append(entity)
                seen_ids.add(entity_id)

                chunk_ids.update(entity["chunk_ids"])
        # Then from global
        if i < len(global_entities):
            entity = global_entities[i]
            entity_id = entity.get("graph_id")
            if entity_id and entity_id not in seen_ids:
                final_entities.append(entity)
                seen_ids.add(entity_id)

                chunk_ids.update(entity["chunk_ids"])

    # Round-robin merge relations
    final_relations = []
    seen_ids = set()

    max_len = max(len(local_relations), len(global_relations))
    for i in range(max_len):
        # First from local
        if i < len(local_relations):
            relation = local_relations[i]
            # Build relation unique identifier
            relation_id = relation.get("graph_id")
            if relation_id not in seen_ids:
                final_relations.append(relation)
                seen_ids.add(relation_id)

                chunk_ids.update(relation["chunk_ids"])

        # Then from global
        if i < len(global_relations):
            relation = global_relations[i]
            # Build relation unique identifier
            relation_id = relation.get("graph_id")
            if relation_id not in seen_ids:
                final_relations.append(relation)
                seen_ids.add(relation_id)

                chunk_ids.update(relation["chunk_ids"])

    # Generate entities context
    entities_context = []
    for i, n in enumerate(final_entities):

        # Get file path from node data
        #file_path = n.get("file_path", "unknown_source")

        entities_context.append(
            {
                "id": i + 1,
                "entity": n["graph_id"],
                "type": n.get("type", "UNKNOWN"),
                "description": n.get("value", "UNKNOWN"),
                #"file_path": file_path,
            }
        )

    # Generate relations context
    relations_context = []
    for i, e in enumerate(final_relations):

        # Get file path from edge data
        #file_path = e.get("file_path", "unknown_source")

        relations_context.append(
            {
                "id": i + 1,
                "entity1": e.get("source"),
                "entity2": e.get("target"),
                "description": e.get("value", "UNKNOWN"),
                #"file_path": file_path,
            }
        )

    # Generate chunks context
    merged_chunks = []
    chunk_ids = [int(id) for id in chunk_ids if str(id).strip().isdigit()]
    for chunk_id in chunk_ids:
        merged_chunks.append(
                            {
                                "content": chunks[chunk_id],
                                #"file_path": chunk.get("file_path", "unknown_source"),
                            }
                        )

    context_data = {
        "entities": entities_context,
        "relations": relations_context,
        "chunks": merged_chunks
    }

    if "entities" in context_data and context_data["entities"]:
        entities = context_data["entities"]
        entities_str = f"--- Entities ---\n{json.dumps(entities, indent=2, ensure_ascii=False)}\n"

    if "relations" in context_data and context_data["relations"]:
        relations = context_data["relations"]
        relations_str = f"--- Relations ---\n{json.dumps(relations, indent=2, ensure_ascii=False)}\n"

    if "chunks" in context_data and context_data["chunks"]:
        chunks = context_data["chunks"]
        chunks_str_list = []
        for chunk in chunks:
            chunks_str_list.append(f"Text: {chunk}")
        chunks_str = f"--- Text Chunks ---\n" + "\n\n".join(chunks_str_list)

    context_str = f"{entities_str}{relations_str}{chunks_str}".strip()

    return context_str