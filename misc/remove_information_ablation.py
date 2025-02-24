import json
import os
import openai
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

def query_openai(message):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[

            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content
def filter_datasets(input_file, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    
    information_categories = {
        "demographic_data": "demographic data, such as age, gender, ethnicity, and occupation",
        "history_taking": "history taking, such as chief complaint, onset of symptom, and duration of symptom",
        "past_history": "past history, such as past medical history, family history, smoking history, alcohol use history, and illicit drug use history",
        "physical_exam": "physical examination results",
        "lab_tests": "lab and diagnostic test findings",
        "others": "demographic data, such as age, gender, ethnicity, and occupation; history taking, such as chief complaint, onset of symptom, and duration of symptom; past history, such as past medical history, family history, smoking history, alcohol use history, and illicit drug use history; physical examination results; lab and diagnostic test findings",
    }

    
    for key, category_desc in information_categories.items():
        output_file = os.path.join(output_folder, f"{key}.json")
        
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump([], f)  

        filtered_data = []
        
        for entry in tqdm(data, desc=f"Processing {key}"):
            question = entry.get("question", "")
            message = f"You are a medical assistant AI, specialized in refining medical questions. Please revise the following question to exclude details related to {category_desc}.\nQuestion: {question}"
            
            refined_question = query_openai(message)  
            new_entry = entry.copy()
            new_entry["question"] = refined_question
            filtered_data.append(new_entry)

            
            with open(output_file, "r+", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_data.append(new_entry)
                f.seek(0)
                json.dump(existing_data, f, indent=4, ensure_ascii=False)

    
    message = f"You are a medical AI trained to refine questions. Remove all questions that do not contain details related to the following: {', '.join(information_categories.values())}."
    
    output_file = os.path.join(output_folder, "others.json")
    
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f)  

    filtered_data_only = []
    for entry in tqdm(data, desc="Filtering dataset with only six categories"):
        question = entry.get("question", "")
        refined_question = query_openai(message + f"\nQuestion: {question}")  
        new_entry = entry.copy()
        new_entry["question"] = refined_question
        filtered_data_only.append(new_entry)

        
        with open(output_file, "r+", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_data.append(new_entry)
            f.seek(0)
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    data_path = "medbullet_dataset.json"
    output_folder_path = "medbullet_ablation/"
    filter_datasets(data_path, output_folder_path)
