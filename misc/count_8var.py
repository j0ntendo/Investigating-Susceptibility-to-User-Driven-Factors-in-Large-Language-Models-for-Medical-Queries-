import json
import pandas as pd
from collections import defaultdict

def process_json(file_path, model_name, include_metadata=True):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"Unexpected JSON structure in {file_path}. Expected a list of entries.")
    
    counts = defaultdict(lambda: {"true": 0, "not_true": 0})
    
    for entry in data:
        if not isinstance(entry, dict):  
            continue
        
        is_correct = entry.get("is_correct")
        
        if model_name == "GPT-4o":
            tone = entry.get("metadata", {}).get("tone", "Unknown")
        else:
            tone = entry.get("label", "Unknown")
        
        ai_role = entry.get("metadata", {}).get("ai_role", "Unknown")
        physician_desc = entry.get("metadata", {}).get("physician_description", "Unknown")
        category = entry.get("category", entry.get("Category", "Unknown"))
        
        metadata_key = (model_name, ai_role, physician_desc, tone, category)
        
        if is_correct is True or is_correct == "true":  
            counts[metadata_key]["true"] += 1
        else:
            counts[metadata_key]["not_true"] += 1
    
    rows = []
    for metadata_key, result in counts.items():
        model_name, ai_role, physician_desc, tone, category = metadata_key
        
        true_count = result["true"]
        not_true_count = result["not_true"]
        
        rows.append({
            "Model Name": model_name,
            "AI Role": ai_role,
            "Physician Description": physician_desc,
            "Tone": tone,
            "Category": category,
            "True": true_count,
            "Not True": not_true_count
        })
    
    return pd.DataFrame(rows)

def save_to_excel(model_files, unperturbed_files, output_excel):
    all_data = []
    unperturbed_data = []
    
    for model_name, json_file in model_files.items():
        try:
            df = process_json(json_file, model_name, include_metadata=True)
            all_data.append(df)
        except Exception as e:
            print(f"❌ Error processing perturbed {model_name}: {e}")
    
    for model_name, json_file in unperturbed_files.items():
        try:
            df = process_json(json_file, model_name, include_metadata=False)
            unperturbed_data.append(df)
        except Exception as e:
            print(f"❌ Error processing unperturbed {model_name}: {e}")
    
    with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_excel(writer, sheet_name="Perturbed_Data", index=False)
        
        if unperturbed_data:
            final_df_unperturbed = pd.concat(unperturbed_data, ignore_index=True)
            final_df_unperturbed.to_excel(writer, sheet_name="Unperturbed_Data", index=False)
        
        workbook = writer.book
        for sheet in ["Perturbed_Data", "Unperturbed_Data"]:
            if sheet in writer.sheets:
                worksheet = writer.sheets[sheet]
                worksheet.set_column('A:A', 20)
                worksheet.set_column('B:F', 30)
                worksheet.set_column('G:H', 12)
    
    print(f"📂 Results saved to {output_excel}")


medbullet = {
    'GPT-4o': 'res_medbullet/gpt/gpt4o_8var.json',
    'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/haiku/c_real_claude_8var.json',
    'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/sonnet/sonnet_8var.json',
    'Gemini 1.5 Pro': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/pro/geminipro_8var.json',
    'Gemini 1.5 Flash': 'res_medbullet/flash/geminiflash_8var.json',
    'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/llama3/llama3.json',
    'Llama 3 Med42 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/medllama3/llama3med.json',
    'DeepSeek (8B)': 'res_medbullet/deepseek/deepseek.json'
}
medqa = {
    'GPT-4o': '/home/ujinkang/Desktop/vscode/ubunut/results/openai/gpt4o_8var.json',
    'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/results/claude/claude_8var.json',
    'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/results/sonnet/c_sonnet_8var.json',
    'Gemini 1.5 Pro': 'results/gemini_pro/geminipro_8var.json',
    'Gemini 1.5 Flash': '/home/ujinkang/Desktop/vscode/ubunut/results/gemini_flash/geminiflash_8var.json',
    'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/llama3/llama3_8var.json',
    'Llama 3 Med42 (8B)': 'results/llama3_med/llama3_med_8var.json',
    'DeepSeek (8B)': 'results/deepseek/deepseek.json'
}
unperturbed_medbullet = { 
    'GPT-4o': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/gpt/ablation/normal.json',
    'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/haiku/ablation/normal.json',
    'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/sonnet/ablation/normal.json',
    'Gemini 1.5 Pro': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/pro/ablation/normal.json',
    'Gemini 1.5 Flash': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/flash/ablation/normal.json',
    'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/llama3/normal.json',
    'Llama 3 Med42 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/medllama3/ablation/normal.json',
    'DeepSeek (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/deepseek/normal.json'
 }  # JSON paths for unperturbed MedBullet
unperturbed_medqa = {
    'GPT-4o': '/home/ujinkang/Desktop/vscode/ubunut/results/openai/test_ablation/no_exclusion.json',
    'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/results/claude/test_ablation/no_exclusion.json',
    'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/results/sonnet/ablation/no_exclusion.json',
    'Gemini 1.5 Pro': '/home/ujinkang/Desktop/vscode/ubunut/results/gemini_pro/ablation/no_exclusion.json',
    'Gemini 1.5 Flash': '/home/ujinkang/Desktop/vscode/ubunut/results/gemini_flash/ablation/no_exclusion.json',
    'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/llama3/test_ablation/no_exclusion.json',
    'Llama 3 Med42 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/llama3_med/ablation/no_exclusion.json',
    'DeepSeek (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/deepseek/ablation/no_exclusion.json'
 }  # JSON paths for unperturbed MedQA

#ave_to_excel(medbullet, unperturbed_medbullet, "new_cat_medbullet_per.xlsx")
save_to_excel(medqa, unperturbed_medqa, "cat_medqa_per.xlsx")


# import json
# import pandas as pd
# from collections import defaultdict

# def process_json(file_path, model_name, include_metadata=True):
#     with open(file_path, 'r') as f:
#         data = json.load(f)
#     if not isinstance(data, list):
#         raise ValueError(f"Unexpected JSON structure in {file_path}. Expected a list of entries.")
#     counts = defaultdict(lambda: {"true": 0, "not_true": 0})
#     for entry in data:
#         if not isinstance(entry, dict):
#             continue
#         is_correct = entry.get("is_correct")
#         if model_name == "GPT-4o":
#             tone = entry.get("metadata", {}).get("tone", "Unknown")
#         else:
#             tone = entry.get("label", "Unknown")
#         ai_role = entry.get("metadata", {}).get("ai_role", "Unknown")
#         physician_desc = entry.get("metadata", {}).get("physician_description", "Unknown")
#         metadata_key = (model_name, ai_role, physician_desc, tone)
#         if is_correct is True or is_correct == "true":
#             counts[metadata_key]["true"] += 1
#         else:
#             counts[metadata_key]["not_true"] += 1
#     rows = []
#     for metadata_key, result in counts.items():
#         model_name, ai_role, physician_desc, tone = metadata_key
#         true_count = result["true"]
#         not_true_count = result["not_true"]
#         rows.append({
#             "Model Name": model_name,
#             "AI Role": ai_role,
#             "Physician Description": physician_desc,
#             "Tone": tone,
#             "True": true_count,
#             "Not True": not_true_count
#         })
#     return pd.DataFrame(rows)

# def save_to_excel(model_files, unperturbed_files, output_excel):
#     all_data = []
#     unperturbed_data = []
#     for model_name, json_file in model_files.items():
#         try:
#             df = process_json(json_file, model_name, include_metadata=True)
#             all_data.append(df)
#         except Exception as e:
#             print(f"❌ Error processing perturbed {model_name}: {e}")
#     for model_name, json_file in unperturbed_files.items():
#         try:
#             df = process_json(json_file, model_name, include_metadata=False)
#             unperturbed_data.append(df)
#         except Exception as e:
#             print(f"❌ Error processing unperturbed {model_name}: {e}")
#     with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
#         if all_data:
#             final_df = pd.concat(all_data, ignore_index=True)
#             final_df.to_excel(writer, sheet_name="Perturbed_Data", index=False)
#         if unperturbed_data:
#             final_df_unperturbed = pd.concat(unperturbed_data, ignore_index=True)
#             final_df_unperturbed.to_excel(writer, sheet_name="Unperturbed_Data", index=False)
#         workbook = writer.book
#         for sheet in ["Perturbed_Data", "Unperturbed_Data"]:
#             if sheet in writer.sheets:
#                 worksheet = writer.sheets[sheet]
#                 worksheet.set_column('A:A', 20)
#                 worksheet.set_column('B:D', 30)
#                 worksheet.set_column('E:F', 12)
#     print(f"📂 Results saved to {output_excel}")

# medbullet = {
#     'GPT-4o': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/gpt/c_gpt4o_8var.json',
#     'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/haiku/c_real_claude_8var.json',
#     'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/sonnet/sonnet_8var.json',
#     'Gemini 1.5 Pro': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/pro/geminipro_8var.json',
#     'Gemini 1.5 Flash': 'res_medbullet/flash/geminiflash_8var.json',
#     'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/llama3/llama3.json',
#     'Llama 3 Med42 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/medllama3/llama3med.json',
#     'DeepSeek (8B)': 'res_medbullet/deepseek/deepseek.json'
# }
# # medqa = {
# #     'GPT-4o': '/home/ujinkang/Desktop/vscode/ubunut/results/openai/gpt4o_8var.json',
# #     'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/results/claude/claude_8var.json',
# #     'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/results/sonnet/c_sonnet_8var.json',
# #     'Gemini 1.5 Pro': 'results/gemini_pro/geminipro_8var.json',
# #     'Gemini 1.5 Flash': '/home/ujinkang/Desktop/vscode/ubunut/results/gemini_flash/geminiflash_8var.json',
# #     'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/llama3/llama3_8var.json',
# #     'Llama 3 Med42 (8B)': 'results/llama3_med/llama3_med_8var.json',
# #     'DeepSeek (8B)': 'results/deepseek/deepseek.json'
# # }
# unperturbed_medbullet = { 
#     'GPT-4o': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/gpt/ablation/normal.json',
#     'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/haiku/ablation/normal.json',
#     'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/sonnet/ablation/normal.json',
#     'Gemini 1.5 Pro': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/pro/ablation/normal.json',
#     'Gemini 1.5 Flash': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/flash/ablation/normal.json',
#     'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/llama3/normal.json',
#     'Llama 3 Med42 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/medllama3/ablation/normal.json',
#     'DeepSeek (8B)': '/home/ujinkang/Desktop/vscode/ubunut/res_medbullet/deepseek/normal.json'
# }
# # unperturbed_medqa = {
# #     'GPT-4o': '/home/ujinkang/Desktop/vscode/ubunut/results/openai/test_ablation/no_exclusion.json',
# #     'Claude 3.5 Haiku': '/home/ujinkang/Desktop/vscode/ubunut/results/claude/test_ablation/no_exclusion.json',
# #     'Claude 3.5 Sonnet': '/home/ujinkang/Desktop/vscode/ubunut/results/sonnet/ablation/no_exclusion.json',
# #     'Gemini 1.5 Pro': '/home/ujinkang/Desktop/vscode/ubunut/results/gemini_pro/ablation/no_exclusion.json',
# #     'Gemini 1.5 Flash': '/home/ujinkang/Desktop/vscode/ubunut/results/gemini_flash/ablation/no_exclusion.json',
# #     'Llama 3 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/llama3/test_ablation/no_exclusion.json',
# #     'Llama 3 Med42 (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/llama3_med/ablation/no_exclusion.json',
# #     'DeepSeek (8B)': '/home/ujinkang/Desktop/vscode/ubunut/results/deepseek/ablation/no_exclusion.json'
# # }
# save_to_excel(medbullet, unperturbed_medbullet, "new_medbullet_per.xlsx")
