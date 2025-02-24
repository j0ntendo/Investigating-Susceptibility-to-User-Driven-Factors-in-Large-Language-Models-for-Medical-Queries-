import os
import json
from tqdm import tqdm

def add_excluded_field(json_file_path):
    """
    Add an 'excluded' field with the file name (excluding 'results ' and the extension) to each entry in the JSON file.

    :param json_file_path: Path to the JSON file to process.
    """
    excluded_info = os.path.splitext(os.path.basename(json_file_path))[0].replace("results ", "")

    with open(json_file_path, 'r', encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {json_file_path}")
            return

    for entry in tqdm(data, desc=f"Processing {os.path.basename(json_file_path)}"):
        entry["excluded"] = excluded_info

    with open(json_file_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print(f"Modified file saved at: {json_file_path}")

def process_file(file_path):
    """
    Process a single JSON file and add an 'excluded' field.

    :param file_path: Path to the JSON file.
    """
    if file_path.endswith(".json"):
        print(f"Processing file: {file_path}")
        add_excluded_field(file_path)
    else:
        print(f"File is not a JSON file: {file_path}")


file_path = "/home/ujinkang/Desktop/vscode/ubunut/results/sonnet/ablation/no_exclusion.json"
process_file(file_path)
