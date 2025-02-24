# import json
# import os
# import pandas as pd
# from collections import defaultdict

# def analyze_is_correct_by_excluded_and_category(folders, output_file):
#     combined_results = []
    
#     for model_name, folder_path in folders.items():
#         results = defaultdict(lambda: defaultdict(lambda: {"true": 0, "other": 0}))
        
#         for filename in os.listdir(folder_path):
#             if filename.endswith(".json"):
#                 file_path = os.path.join(folder_path, filename)
#                 with open(file_path, 'r') as file:
#                     try:
#                         data = json.load(file)
#                         for entry in data:
#                             excluded = entry.get("excluded", "Unknown")
#                             category = entry.get("category", "Unknown")
#                             is_correct = entry.get("is_correct", "other")

#                             if is_correct is True or is_correct == "true":
#                                 results[excluded][category]["true"] += 1
#                             else:
#                                 results[excluded][category]["other"] += 1
#                     except json.JSONDecodeError:
#                         print(f"Error decoding JSON in file: {filename}")
        
#         for excluded, category_dict in results.items():
#             for category, counts in category_dict.items():
#                 true_count = counts["true"]
#                 other_count = counts["other"]
#                 total = true_count + other_count
#                 percentage = (true_count / total * 100) if total > 0 else 0
#                 combined_results.append([model_name, excluded, category, true_count, other_count, percentage])
    
#     df = pd.DataFrame(combined_results, columns=["Model", "Excluded", "Category", "True", "Other", "Percentage Correct"])
#     df.to_excel(output_file, sheet_name="Results", index=False)

# medbullet = {
#     "GPT-4o": "res_medbullet/gpt/ablation",
#     "Claude 3.5 Haiku": "res_medbullet/haiku/ablation",
#     "Claude 3.5 Sonnet": "res_medbullet/sonnet/ablation",
#     "Gemini 1.5 Flash": "res_medbullet/flash/ablation",
#     "Gemini 1.5 Pro": "res_medbullet/pro/ablation",
#     "Llama 3 (8B)": "res_medbullet/llama3/ablation",
#     "Llama 3 Med42 (8B)": "res_medbullet/medllama3/ablation",
#     "DeepSeek (8B)": "res_medbullet/deepseek/ablation"
# }

# medqa = {
#     "GPT-4o": "results/openai/test_ablation",
#     "Claude 3.5 Haiku": "results/claude/test_ablation",
#     "Claude 3.5 Sonnet": "/home/ujinkang/Desktop/vscode/ubunut/results/sonnet/ablation",
#     "Gemini 1.5 Flash": "results/gemini_flash/ablation",
#     "Gemini 1.5 Pro": "results/gemini_pro/ablation",
#     "Llama 3 (8B)": "results/llama3/test_ablation",
#     "Llama 3 Med42 (8B)": "results/llama3_med/ablation",
#     "DeepSeek (8B)": "results/deepseek/ablation"
# }

 

# # Call the function to analyze JSON files from different folders and create two Excel files
# analyze_is_correct_by_excluded_and_category(medbullet, "medbullet_ablation.xlsx")
# #analyze_is_correct_by_excluded_and_category(medqa, "medqa_ablation.xlsx")

import json
import os
from collections import defaultdict

def analyze_is_correct_by_excluded_and_category(folders):
    for model_name, folder_path in folders.items():
        results = defaultdict(lambda: defaultdict(lambda: {"true": 0, "other": 0}))
        
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r') as file:
                    try:
                        data = json.load(file)
                        for entry in data:
                            excluded = entry.get("excluded", "Unknown")
                            category = entry.get("category", "Unknown")
                            is_correct = entry.get("is_correct", "other")

                            if is_correct is True or is_correct == "true":
                                results[excluded][category]["true"] += 1
                            else:
                                results[excluded][category]["other"] += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {filename}")

        print(f"\nModel: {model_name}")
        for excluded, category_dict in results.items():
            for category, counts in category_dict.items():
                true_count = counts["true"]
                other_count = counts["other"]
                print(f"Excluded: {excluded}, Category: {category}, True: {true_count}, Other: {other_count}")

json_folders = {
        "deepseek": "/home/ujinkang/Desktop/vscode/ubunut/results/deepseek/ablation",
        "llama3": "/home/ujinkang/Desktop/vscode/ubunut/results/llama3/test_ablation",
        "llama3_med": "/home/ujinkang/Desktop/vscode/ubunut/results/llama3_med/ablation",
        "medbullet_llama3": "/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/llama3/ablation",
        "medbullet_deepseek": "/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/deepseek/ablation"
}

analyze_is_correct_by_excluded_and_category(json_folders)
