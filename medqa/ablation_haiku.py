import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import anthropic
import re

load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key)

def query_claude(prompt):
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.content[0].text

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_answer(response_text):
    response_clean = response_text.strip()
    match = re.search(r"Answer:\s*\((\w)\)\s*(.+)", response_clean)
    if match:
        return match.group(2).strip()
    first_line = response_clean.split("\n", 1)[0]
    return first_line.strip()

def create_prompt(data_entry):
    question = data_entry["question"]
    options = data_entry["options"]
    prompt = (
        f"You are a medical AI. Below is a medical question followed by five possible options.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        + "\n".join([f"({key}) {value}" for key, value in options.items()])
        + "\n\nAlways respond by outputting your answer first and then your explanation"
    )
    return prompt

def save_results_to_json(results, output_file):
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

def process_datasets(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json") and file_name != "other.json":
            input_file = os.path.join(folder_path, file_name)
            output_file = os.path.join(output_folder, file_name)
            try:
                dataset = load_json(input_file)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading {input_file}: {e}")
                continue
            results = []
            for data_entry in tqdm(dataset, desc=f"Processing {input_file}"):
                prompt = create_prompt(data_entry)
                try:
                    response_text = query_claude(prompt)
                    chosen_answer = extract_answer(response_text)
                    is_correct = chosen_answer == data_entry["answer"]
                    record = {
                        "prompt": prompt,
                        "response": response_text,
                        "chosen_answer": chosen_answer,
                        "correct_answer": data_entry["answer"],
                        "answer_idx": data_entry.get("answer_idx", ""),
                        "category": data_entry["Category"],
                        "is_correct": is_correct,
                    }
                    results.append(record)
                except Exception as e:
                    print(f"Error processing {input_file}: {e}")
                    continue
            save_results_to_json(results, output_file)

def main():
    INPUT_FOLDER = "" # folder path containing the json files
    OUTPUT_FOLDER = "results/claude/test_ablation"
    process_datasets(INPUT_FOLDER, OUTPUT_FOLDER)

if __name__ == "__main__":
    main()