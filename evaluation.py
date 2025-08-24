# Evaluation by comparing two models
# Dataset: Mix from UltraDomain
# https://huggingface.co/datasets/TommyChien/UltraDomain/resolve/main/mix.jsonl?download=true
from dotenv import load_dotenv
import os

load_dotenv(".env")

api_key = os.getenv("DEEPSEEK_API_KEY")


from langchain_openai import ChatOpenAI

from datasets import load_dataset
import html
import json
import json_repair
import jsonlines

from simplified_lightrag import simplified_lightrag
from naive_rag import naive_rag
from prompt import *
import utils

model_name1 = "naive_rag"
model_name2 = "simplified_lightrag"
output_file_path = "evaluation_responses.json"
query_mode = ""

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com/v1"
)

dataset = load_dataset("json", data_files={"train": "mix.jsonl"})

inputs = dataset["train"]["input"][:3]
contexts = dataset["train"]["context"][:3]

prompts = []
for i, (input, context) in enumerate(zip(inputs, contexts)):
    answer1 = naive_rag(input, context)
    answer2 = simplified_lightrag(input, context, query_mode)
    prompt_template = PROMPTS["evaluation"]
    prompt = prompt_template.format(
        query=input,
        answer1=answer1,
        answer2=answer2
    )
    prompts.append(prompt)
    print(f"{i}/{len(inputs)}")


responses = llm.batch(prompts)

win_counts = {
        'Comprehensiveness': {"Answer 1": 0, "Answer 2": 0},
        'Diversity': {"Answer 1": 0, "Answer 2": 0},
        'Empowerment': {"Answer 1": 0, "Answer 2": 0},
        'Overall': {"Answer 1": 0, "Answer 2": 0}
    }

total = 0
for response in responses:
    try:
        data = json_repair.loads(response)
        
        win_counts['Comprehensiveness'][utils.clean_str(data['Comprehensiveness']['Winner'])] += 1
        win_counts['Diversity'][utils.clean_str(data['Diversity']['Winner'])] += 1
        win_counts['Empowerment'][utils.clean_str(data['Empowerment']['Winner'])] += 1
        win_counts['Overall'][utils.clean_str(data['Overall Winner']['Winner'])] += 1
        count += 1

        with jsonlines.open(output_file_path, mode="w") as writer:
            writer.write(response)
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"JSON parsing error: {e}")
        print(f"LLM response: {response}")

for category, counts in win_counts.items():
    print(win_counts[category])
    win_rates = {}
    win_rates[category] = {
        model_name1: f"{(counts["Answer 1"] / total) * 100:.1f}%",
        model_name2: f"{(counts["Answer 2"] / total) * 100:.1f}%"
    }
    print(f"{category}: {win_rates[category]}")