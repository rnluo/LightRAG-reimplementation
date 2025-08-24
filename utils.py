# The functions in this file are borrowed from:
# Project: LightRAG 
# Sources: 
#   https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py
#   https://github.com/HKUDS/LightRAG/blob/main/lightrag/utils.py
# Author: HKUDS
# License: MIT

from langchain_openai import ChatOpenAI

from typing import Any
import html
import os
import pickle
import re

from prompt import *

# Output parsing
def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    content = content if content is not None else ""
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]

def normalize_extracted_info(name: str, is_entity=False) -> str:
    """Normalize entity/relation names and description with the following rules:
    1. Remove spaces between Chinese characters
    2. Remove spaces between Chinese characters and English letters/numbers
    3. Preserve spaces within English text and numbers
    4. Replace Chinese parentheses with English parentheses
    5. Replace Chinese dash with English dash
    6. Remove English quotation marks from the beginning and end of the text
    7. Remove English quotation marks in and around chinese
    8. Remove Chinese quotation marks

    Args:
        name: Entity name to normalize

    Returns:
        Normalized entity name
    """
    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash
    name = name.replace("—", "-").replace("－", "-")

    # Use regex to remove spaces between Chinese characters
    # Regex explanation:
    # (?<=[\u4e00-\u9fa5]): Positive lookbehind for Chinese character
    # \s+: One or more whitespace characters
    # (?=[\u4e00-\u9fa5]): Positive lookahead for Chinese character
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove spaces between Chinese and English/numbers/symbols
    name = re.sub(
        r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name
    )
    name = re.sub(
        r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name
    )

    # Remove English quotation marks from the beginning and end
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        name = name[1:-1]
    if len(name) >= 2 and name.startswith("'") and name.endswith("'"):
        name = name[1:-1]

    if is_entity:
        # remove Chinese quotes
        name = name.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
        # remove English queotes in and around chinese
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)

    return name

def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

def is_float_regex(value: str) -> bool:
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

# Single entity/relation extraction
def _handle_single_entity_extraction(
    record_attributes: list[str],
    #chunk_key: str,
    #file_path: str = "unknown_source",
    llm: ChatOpenAI
):
    if len(record_attributes) < 4 or '"entity"' not in record_attributes[0]:
        return None
    entity_name = clean_str(record_attributes[1]).strip()
    if not entity_name:
        return None
    entity_name = normalize_extracted_info(entity_name, is_entity=True)
    if not entity_name or not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2]).strip('"')
    if not entity_type.strip() or entity_type.startswith('("'):
        return None
    
    entity_description = clean_str(record_attributes[3])
    entity_description = normalize_extracted_info(entity_description)
    if not entity_description.strip():
        return None
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        #source_id=chunk_key,
        #file_path=file_path,
    )
    
def _handle_single_relation_extraction(
    record_attributes: list[str],
    #chunk_key: str,
    #file_path: str = "unknown_source",
    llm
):
    if len(record_attributes) < 5 or '"relationship"' not in record_attributes[0]:
        return None
    # add this record as edge
    source = clean_str(record_attributes[1])
    target = clean_str(record_attributes[2])
    
    source = normalize_extracted_info(source, is_entity=True)
    target = normalize_extracted_info(target, is_entity=True)
    if not source or not source.strip():
        return None

    if not target or not target.strip():
        return None

    if source == target:
        return None

    edge_description = clean_str(record_attributes[3])
    edge_description = normalize_extracted_info(edge_description)

    edge_keywords = normalize_extracted_info(
        clean_str(record_attributes[4]), is_entity=True
    )
    edge_keywords = edge_keywords.replace("，", ",")

    #edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1].strip('"').strip("'"))
        if is_float_regex(record_attributes[-1].strip('"').strip("'"))
        else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        #source_id=edge_source_id,
        #file_path=file_path,
    )

# Summarization and merging
def _handle_entity_relation_summary(
    entity_or_relation_names: list[str],
    descriptions: list[str],
    filename: str,
    llm: ChatOpenAI
    ):
    prompts = []
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    for entity_or_relation_name, description in zip(entity_or_relation_names, descriptions):
        prompt = prompt_template.format(
            entity_name=entity_or_relation_name,
            description_list=description,
            language=PROMPTS["DEFAULT_LANGUAGE"]
        )
        prompts.append(prompt)

    cache_file = f"{filename} summaries_cache.pkl"
    
    if filename:
        if os.path.exists(cache_file):
            print("Loading summaries from cache...")
            with open(cache_file, 'rb') as f:
                summaries = pickle.load(f)
        else:
            print("Fetching summaries from LLM and caching...")
            summaries = llm.batch(prompts)
            with open(cache_file, 'wb') as f:
                pickle.dump(summaries, f)
    else:
        summaries = llm.batch(prompts) 
    return summaries