from dotenv import load_dotenv
import json
import os
import re
import ollama
from tqdm import tqdm

load_dotenv()

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_results_to_json(results, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_results = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: {output_file} is corrupted or empty. Starting fresh.")
                existing_results = []
    else:
        existing_results = []

    existing_prompts = {entry["prompt"] for entry in existing_results}
    new_results = [record for record in results if record["prompt"] not in existing_prompts]
    
    combined_results = existing_results + new_results
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4)
    
    print(f"Results saved to {output_file}")

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
    return answer_match.group(1) if answer_match else "Unknown"

def query_ollama(messages):
    response = ollama.generate(model="llama3:instruct", prompt=messages, stream=False)
    return response["response"]

def process_dataset(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return
    
    dataset = load_json(input_file)
    results = []
    
    for data_entry in tqdm(dataset, desc="Processing dataset"):
        try:
            prompt = create_prompt(data_entry)
            response_text = query_ollama(prompt)
            model_answer = extract_answer(response_text)
            
            record = {
                "prompt": prompt,
                "response": response_text,
                "correct_answer": data_entry["answer"],
                "answer_idx": data_entry["answer_idx"],
                "category": data_entry["category"],
                "is_correct": model_answer == data_entry["answer"],
            }
            results.append(record)
            
            save_results_to_json([record], output_file)
        except Exception as e:
            print(f"Error during query for {input_file}: {e}")
            continue

if __name__ == "__main__":
    INPUT_FILE = "/home/ujinkang/Desktop/vscode/ubunut/medbullet_dataset.json"
    OUTPUT_FILE = "medbullet_res/llama3_ablation.json"
    process_dataset(INPUT_FILE, OUTPUT_FILE)