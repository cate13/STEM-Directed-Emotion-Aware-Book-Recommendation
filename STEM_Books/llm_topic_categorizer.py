import os

HF_TOKEN = ""

os.environ["HF_HOME"] = "/data/ccate13/data_cleaner/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/ccate13/data_cleaner/hf_cache"

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import csv
import torch
import gc
from tqdm import tqdm

def message_maker(topic):
    prompt = (
        "You are a helpful assistant that classifies topics. "
        "Decide if the topic is about STEM (Science, Technology, Engineering, or Mathematics). "
        "Answer only 'STEM' or 'Not STEM'.\n"
        f"Description: {topic}"
    )
    return [{"role": "user", "content": prompt}]


def individual_model_eval(tokenizer, topic, pipe):
    messages = message_maker(topic)

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    output = pipe(
        formatted_prompt,
        max_new_tokens=5,
        do_sample=False,
        temperature=0.0
    )

    response_text = output[0]["generated_text"][len(formatted_prompt):].strip()
    return response_text


model_ids = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct"
]

path = "categories.txt"

# Read topics once
with open(path, encoding="utf-8") as f:
    topics = [line.strip() for line in f]

results = {"Topic": topics}

# 🔥 Outer progress bar (models)
for model_id in tqdm(model_ids, desc="Models", position=0):

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=HF_TOKEN
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=HF_TOKEN,
        device_map="auto",
        torch_dtype=torch.float16
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )

    model_results = []

    # 🔥 Inner progress bar (topics)
    for topic in tqdm(
        topics,
        desc=f"{model_id}",
        leave=False,
        position=1
    ):
        response = individual_model_eval(tokenizer, topic, pipe)
        ans = response.lower().replace(".", "").strip()

        if ans.startswith("stem"):
            pred_is_stem = True
        elif ans.startswith("not stem"):
            pred_is_stem = False
        else:
            pred_is_stem = None

        model_results.append(pred_is_stem)

    results[model_id] = model_results

    # Free GPU memory
    del model
    del pipe
    torch.cuda.empty_cache()
    gc.collect()


# Write CSV
with open("topic_eval_all_models.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    header = list(results.keys())
    writer.writerow(header)

    for i in range(len(topics)):
        row = [results[key][i] for key in header]
        writer.writerow(row)

print("Done.")