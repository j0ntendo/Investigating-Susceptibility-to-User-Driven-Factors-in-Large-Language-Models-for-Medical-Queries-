import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
import openai
from multiprocessing import Pool

load_dotenv()

def query_openai(message):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "developer", "content": "You are a medical AI trained to evaluate answers on the MedQA dataset. Read the full LLM response carefully, including both the answer and its explanation. Compare both components against the correct answer and explanation criteria. Output 'true' only if both the answer and the explanation are correct and fully consistent with the expected solution; otherwise, output 'false'. Do not provide any additional text or explanations."},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "false"

def process_json_file(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return 0

    corrected_count = 0
    
    for entry in tqdm(data, desc=f"Validating {os.path.basename(json_path)}", unit="entry"):
        if entry.get('is_correct') != False:
            response = entry.get("response", "")
            correct_answer = entry.get("correct_answer", "")
            answer_idx = entry.get("answer_idx", "")

            query_message = (
                f"LLM Response: '{response}'\n"
                f"Correct Answer: '{correct_answer}'\n"
                f"Correct Answer Index: '{answer_idx}'"
            )

            is_correct = query_openai(query_message)
            if is_correct == 'true':
                entry["is_correct"] = True
                corrected_count += 1

    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
    
    print(f"Updated file: {json_path} | Corrected: {corrected_count}")
    return corrected_count

def process_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return 0

    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    if not json_files:
        print(f"No JSON files found in: {folder_path}")
        return 0

    total_corrected = 0
    for json_file in json_files:
        total_corrected += process_json_file(os.path.join(folder_path, json_file))
    
    print(f"Total corrected for {os.path.basename(folder_path)}: {total_corrected}")
    return total_corrected

def main():

    
    json_folders = [
        "res_medbullet/sonnet/ablation",
        "res_medbullet/pro/ablation",
        "res_medbullet/medllama3/ablation",
        "res_medbullet/llama3/ablation",
        "res_medbullet/haiku/ablation",
        "res_medbullet/gpt/ablation",
        "res_medbullet/flash/ablation",
        "res_medbullet/deepseek/ablation"
    ]
    
    results = {}
    with Pool(processes=2) as pool:
        results_list = pool.map(process_folder, json_folders)
    
    for model, count in zip(json_folders, results_list):
        results[model] = count
    
    print("\nFinal Corrections Per Model:")
    for model, count in results.items():
        print(f"{model}: {count}")

if __name__ == "__main__":
    main()

