import json
import os
import multiprocessing
from tqdm import tqdm
from dotenv import load_dotenv
import openai

load_dotenv()

def query_openai(message):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.0,
            messages=[
                {"role": "developer", "content": "You are a medical AI trained to evaluate answers on a medical QA dataset. Read the entire LLM response carefully, including all parts of the output such as the answer and the explanation. Do not only focus on the first section; the whole output must be taken into account. Compare both components against the correct answer and explanation criteria. Output 'true' only if both the answer and the explanation are correct and fully consistent with the expected solution; otherwise, output 'false'. Do not provide any additional text or explanations.\n\nBelow is an example where even though the answer in the first part was right, the explanation said that it wasnt the correct answer:\n\nInput:\nAnswer: C) Bromocriptine\n\nExplanation: According to the question, other novice physicians have already ruled out Bromocriptine. Therefore, we are left with the remaining options to find the correct answer. The symptoms described include abnormal discharge from both nipples (galactorrhea) and diminished sexual drive, which are known side effects of certain antipsychotic medications.\n\nMetoclopramide (A), Haloperidol (B), and Fluphenazine (D) can all cause galactorrhea and decreased libido as adverse effects. Risperidone (E) is also a possibility, although less likely to cause galactorrhea specifically.\n\nHowever, since Bromocriptine was already ruled out, it cannot be the culprit.\n\nOutput: false"},
                {"role": "user", "content": message}
            ]
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "false"

def process_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    corrected_count = 0
    for entry in tqdm(data, desc=f"Processing {os.path.basename(json_path)}", unit="entry"):
        response_text = entry.get("response", "")
        correct_answer = entry.get("answer", "")
        answer_idx = entry.get("answer_idx", "")
        query_message = (
            f"LLM Response: '{response_text}'\n"
            f"Correct Answer: '{correct_answer}'\n"
            f"Correct Answer Letter: '{answer_idx}'"
        )
        is_correct = query_openai(query_message)
        if is_correct == "true":
            if not entry.get("is_correct", False):
                corrected_count += 1
            entry["is_correct"] = True
        else:
            entry["is_correct"] = False
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    print(f"Processed file updated: {json_path} — Corrected {corrected_count} entries")
    return json_path, corrected_count

def main():
    json_files = [
        #"/home/ujinkang/Desktop/vscode/ubunut/results/sonnet/c_sonnet_8var.json",
        #"/home/ujinkang/Desktop/vscode/ubunut/results/openai/gpt4o_8var.json",
        "/home/ujinkang/Desktop/vscode/ubunut/results/llama3_med/llama3_med_8var.json",
        "/home/ujinkang/Desktop/vscode/ubunut/results/llama3/llama3_8var.json",
        #"/home/ujinkang/Desktop/vscode/ubunut/results/gemini_pro/geminipro_8var.json",
        #"/home/ujinkang/Desktop/vscode/ubunut/results/gemini_flash/geminiflash_8var.json",
        "/home/ujinkang/Desktop/vscode/ubunut/results/deepseek/deepseek.json",
        #"/home/ujinkang/Desktop/vscode/ubunut/results/claude/claude_8var.json"
    ]


    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(process_json_file, json_files)
    
    total_corrections = 0
    print("\nSummary of corrections:")
    for file_path, count in results:
        print(f"{file_path}: {count} entries corrected")
        total_corrections += count
    print(f"\nTotal entries corrected across all files: {total_corrections}")

if __name__ == "__main__":
    main()
