from dotenv import load_dotenv
import json
import openai
from tqdm import tqdm
import os
import re

load_dotenv()

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_results_to_json(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_path = os.path.join(output_dir, filename)

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {filename} is corrupted or empty. Starting fresh.")
                existing_results = []
    else:
        existing_results = []

    existing_prompts = {entry["prompt"] for entry in existing_results}
    new_results = [record for record in results if record["prompt"] not in existing_prompts]

    combined_results = existing_results + new_results

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    print(f"Results saved to {out_path}")

def extract_answer(response_text):
    response_clean = response_text.strip()
    
    match = re.search(r"\(([A-Z])\)\s*(.+?)(?=\n\n|$)", response_clean, re.MULTILINE)
    if match:
        return match.group(2).strip()  

    first_line = response_clean.split("\n\n", 1)[0]
    return first_line.strip()

def create_prompt(data_entry):
    question = data_entry["question"]
    options = data_entry["options"]
    prompt = (
        f"You are a medical AI. Below is a medical question followed by five possible options.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        + "\n".join([f"({key}) {value}" for key, value in options.items()])
        + "\n\nAlways respond by outputting the answer first and then your explanation"
    )
    return prompt

def query_openai(message):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o", max_completion_tokens=250, messages=[
            {"role": "user", "content": message} 
        ]
    )
    return response.choices[0].message.content

def process_datasets(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    dataset_files = [f for f in os.listdir(input_dir) if f.endswith(".json") and f != "other.json"]
    if not dataset_files:
        print(f"No JSON files found in '{input_dir}'.")
        return

    for dataset_file in tqdm(dataset_files, desc="Processing datasets"):
        dataset_path = os.path.join(input_dir, dataset_file)
        dataset_name = os.path.splitext(dataset_file)[0]  

        dataset = load_json(dataset_path)
        output_file = f"{dataset_name}_results.json"
        results = []

        for data_entry in tqdm(dataset, desc=f"Processing {dataset_file}", leave=False):
            try:
                prompt = create_prompt(data_entry)
                response_text = query_openai(prompt)
                model_answer = extract_answer(response_text)

                record = {
                    "prompt": prompt,
                    "response": response_text,
                    "model_answer": model_answer,
                    "correct_answer": data_entry["answer"],
                    "answer_idx": data_entry["answer_idx"],
                    "category": data_entry.get("Category", ""),
                    "is_correct": model_answer == data_entry["answer"],
                }
                results.append(record)

                save_results_to_json([record], output_dir, output_file)
            
            except Exception as e:
                print(f"Error during query for {dataset_file}: {e}")
                continue

def main():
    INPUT_DIR = ""  # folder path including the json files
    OUTPUT_DIR = "/home/ujinkang/Desktop/vscode/ubunut/results/openai/test_ablation"  

    process_datasets(INPUT_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()