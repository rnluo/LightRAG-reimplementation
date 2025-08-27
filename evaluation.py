# Evaluation by comparing two models with another llm
# Dataset: Mix from UltraDomain
# https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/mix.jsonl?download=true
from dotenv import load_dotenv
import os

load_dotenv(".env")
api_key = os.getenv("DEEPSEEK_API_KEY")

import asyncio
import json
import json_repair
import jsonlines

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import openai_complete
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

from datasets import load_dataset
from langchain_openai import ChatOpenAI

from simplified_lightrag import simplified_lightrag
from naive_rag import naive_rag
from prompt import *
import utils


# Initialize LightRAG from the working directory
async def initialize_rag():
    rag = LightRAG(
        working_dir=".",
        llm_model_func=openai_complete,
        llm_model_name=os.getenv("LLM_MODEL", "deepseek-chat"),
        summary_max_tokens=8192,
        llm_model_kwargs={
            "base_url": os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com/v1"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "timeout": int(os.getenv("TIMEOUT", "1800")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3:567m"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag

# Call LightRAG for generation
async def lightrag(
        input: str,
        context: str,
        rag: LightRAG
):
    if rag is None:
        rag = await initialize_rag()

    await rag.ainsert(context)

    answer_chunks = []
    chunks = await rag.aquery(
        input,
        param=QueryParam(mode='hybrid', stream=True, enable_rerank=False),
    )
    async for chunk in chunks:
        answer_chunks.append(chunk)
    answer = "".join(answer_chunks)

    return answer

# Set model name in output
model_name1 = "naive_rag"
#model_name1 = "lightrag"
model_name2 = "simplified_lightrag"

# Set output path
output_file_path = "evaluation_responses.json"
match_results_path = "evaluation_results.json"
query_mode = "hybrid"

rag = None

async def main():
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
        
    # Load dataset from the working directory
    dataset = load_dataset("json", data_files={"train": "mix.jsonl"})
    inputs = dataset["train"]["input"][:50]
    contexts = dataset["train"]["context"][:50]

    # Generate prompts
    prompts = []
    for i, (input, context) in enumerate(zip(inputs, contexts)):
        answer1 = naive_rag(input, context)
        #answer1 = await lightrag(input, context, rag)
        print(f"{i+1}/{len(inputs)}")
        answer2 = simplified_lightrag(input, context, query_mode)
        print(f"{i+1}/{len(inputs)}")
        prompt_template = PROMPTS["evaluation"]
        prompt = prompt_template.format(
            query=input,
            answer1=answer1,
            answer2=answer2
        )

    # Generate llm evaluations
    responses = await llm.abatch(prompts)

    # Parse and record llm evaluations
    win_counts = {
            'Comprehensiveness': {"Answer 1": 0, "Answer 2": 0},
            'Diversity': {"Answer 1": 0, "Answer 2": 0},
            'Empowerment': {"Answer 1": 0, "Answer 2": 0},
            'Overall': {"Answer 1": 0, "Answer 2": 0}
        }
    total = 0
    for response in responses:
        try:
            data = json_repair.loads(response.content)
            
            win_counts['Comprehensiveness'][utils.clean_str(data['Comprehensiveness']['Winner'])] += 1
            win_counts['Diversity'][utils.clean_str(data['Diversity']['Winner'])] += 1
            win_counts['Empowerment'][utils.clean_str(data['Empowerment']['Winner'])] += 1
            win_counts['Overall'][utils.clean_str(data['Overall Winner']['Winner'])] += 1
            total += 1

            with jsonlines.open(output_file_path, mode="a") as writer:
                writer.write(f"{response.content}\n")
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"JSON parsing error: {e}")
            print(f"LLM response: {response}")

    # Display and save the results
    for category, counts in win_counts.items():
        print(win_counts[category])
        win_rates = {}
        win_rates[category] = {
            model_name1: f"{(counts["Answer 1"] / total) * 100:.1f}%",
            model_name2: f"{(counts["Answer 2"] / total) * 100:.1f}%"
        }
        print(f"{category}: {win_rates[category]}")

        with jsonlines.open(match_results_path, mode="a") as writer:
                writer.write(f"{win_rates[category]}\n")

if __name__ == "__main__":
    asyncio.run(main())