import json
import os
from dotenv import load_dotenv
from tqdm import tqdm
import anthropic
import re

# Load environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=anthropic_api_key)

def query_claude(prompt):
    """Query Claude AI model with a given prompt."""
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=350,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    return response.content[0].text

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_answer(response_text):
    """Extract the answer from the model's response."""
    response_clean = response_text.strip()
    
    # Match pattern "Answer: (X) Answer text"
    match = re.search(r"Answer:\s*\((\w)\)\s*(.+)", response_clean)
    
    if match:
        return match.group(2).strip()  # Extract text after the option letter

    # Fallback: use the first line if no pattern matches
    first_line = response_clean.split("\n", 1)[0]
    return first_line.strip()

def create_prompt(data_entry):
    """Generate a prompt for Claude AI based on the question and options."""
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
    """Save processed results to a JSON file."""
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")

def process_dataset(file_path, output_file):
    """Process a single JSON dataset file."""
    try:
        dataset = load_json(file_path)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding {file_path}: {e}")
        return

    results = []

    for data_entry in tqdm(dataset, desc=f"Processing {os.path.basename(file_path)}"):
        prompt = create_prompt(data_entry)
        try:
            response_text = query_claude(prompt)
            chosen_answer = extract_answer(response_text)
            answer = data_entry["answer"]
            is_correct = chosen_answer == answer

            record = {
                "prompt": prompt,
                "response": response_text,
                "chosen_answer": chosen_answer,
                "correct_answer": data_entry["answer"],
                "answer_idx": data_entry.get("answer_idx", ""),
                "category": data_entry["category"],
                "is_correct": is_correct,
            }
            results.append(record)
        except Exception as e:
            print(f"Error during query or processing for {file_path}: {e}")
            continue

    save_results_to_json(results, output_file)

def main():
    """Main function to process multiple JSON files in a folder."""
    INPUT_DIR = "/home/ujinkang/Desktop/vscode/ubunut/medbullet_asfd"
    OUTPUT_DIR = "/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/sonnet/ablation"
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return
    
    json_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".json")]
    
    if not json_files:
        print(f"No JSON files found in '{INPUT_DIR}'.")
        return

    for json_file in tqdm(json_files, desc="Processing datasets"):
        input_path = os.path.join(INPUT_DIR, json_file)
        output_path = os.path.join(OUTPUT_DIR, json_file)
        process_dataset(input_path, output_path)

if __name__ == "__main__":
    main()
