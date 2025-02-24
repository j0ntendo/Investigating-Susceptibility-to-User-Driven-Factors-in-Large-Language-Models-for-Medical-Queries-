

from dotenv import load_dotenv
import json
import os
from tqdm import tqdm
import re
import ollama

load_dotenv()

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_results_to_json(results, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_path = os.path.join(output_dir, filename)

    # Read existing results from the file, if any
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {filename} is corrupted or empty. Starting fresh.")
                existing_results = []
    else:
        existing_results = []

    # Deduplicate by ensuring no repeated prompts
    existing_prompts = {entry["prompt"] for entry in existing_results}
    new_results = [record for record in results if record["prompt"] not in existing_prompts]

    # Append only new results
    combined_results = existing_results + new_results

    # Write combined results to the file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)

    print(f"Results saved to {out_path}")


def create_prompt(data_entry):
    question = data_entry["question"]
    options = data_entry["options"]
    prompt = (
        f"You are a medical AI. Below is a medical question followed by five possible options. Answer the question.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        + "\n".join([f"({key}) {value}" for key, value in options.items()])
        + "\nAlways output your answer in this format: Answer: [your answer] Explanation:[your explanation]"
    )
    return prompt

def extract_answer(response_text):
    response_clean = response_text.strip()
    answer_match = re.search(r"Answer:\s*([A-Z])", response_clean)
    if answer_match:
        return answer_match.group(1) 
    return "Unknown"


def query_ollama(messages):
    response = ollama.generate(model="deepseek-r1:8b", prompt=messages, stream=False)
    return response["response"]



def process_datasets(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        return

    dataset_files = [f for f in os.listdir(input_dir) if f.endswith(".json") and f != "other.json"]
    if not dataset_files:
        print(f"No valid JSON files found in '{input_dir}'.")
        return

    for dataset_file in tqdm(dataset_files, desc="Processing datasets"):
        dataset_path = os.path.join(input_dir, dataset_file)
        dataset_name = os.path.splitext(dataset_file)[0]  # Remove extension for naming output file

        dataset = load_json(dataset_path)
        output_file = f"{dataset_name}.json"
        results = []

        for data_entry in tqdm(dataset, desc=f"Processing {dataset_file}", leave=False):
            try:
                prompt = create_prompt(data_entry)
                response_text = query_ollama(prompt)
                model_answer = extract_answer(response_text)

                record = {
                    "prompt": prompt,
                    "response": response_text,
                    "correct_answer": data_entry["answer"],
                    "answer_idx": data_entry["answer_idx"],
                    "category": data_entry.get("Category", ""),
                    "is_correct": model_answer == data_entry["answer"],
                }
                results.append(record)

                # Save results incrementally
                save_results_to_json([record], output_dir, output_file)

            except Exception as e:
                print(f"Error during query for {dataset_file}: {e}")
                continue



if __name__ == "__main__":
    DATA_DIR = ""   # folder path including the json files
    OUTPUT_DIR = "results/deepseek/ablation"
    process_datasets(DATA_DIR, OUTPUT_DIR)
