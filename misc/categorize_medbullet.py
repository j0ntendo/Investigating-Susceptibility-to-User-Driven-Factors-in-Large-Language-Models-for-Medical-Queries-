import orjson
from dotenv import load_dotenv
import openai
from tqdm import tqdm
import os
import json

load_dotenv()

def query_openai_for_classification(message):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=openai_api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[
            {"role": "system", "content": """
            You are a medical exam question classifier. You have exactly three categories to choose from. 
            Each category has the following detailed descriptions:

            1) Diagnosis:
               - Identifying diseases or conditions based on patient history, physical examination, 
                 and laboratory or diagnostic test findings.
               - Recognizing signs and symptoms of diseases.
               - Selecting and interpreting laboratory and diagnostic studies.
               - Formulating differential diagnoses and selecting the most likely diagnosis.
               - Determining prognosis or patient outcomes based on clinical and diagnostic information.

            2) Pharmacotherapy, Interventions and Management:
               - Developing a plan to treat or manage a patient's condition using pharmacological 
                 and non-pharmacological methods.
               - Selecting appropriate drugs, determining doses, and monitoring drug efficacy and side effects.
               - Recognizing contraindications, drug interactions, and adverse effects.
               - Planning and monitoring surgical or non-surgical interventions.
               - Managing acute and chronic conditions, including emergency care and follow-ups.

            3) Health Maintenance, Prevention and Surveillance:
               - Promoting health and preventing disease through risk assessment, screening, and 
                 patient education.
               - Identifying risk factors for disease and selecting appropriate preventive measures 
                 (e.g., vaccinations, lifestyle changes).
               - Implementing screening programs and interpreting screening results.
               - Educating patients about health maintenance and long-term surveillance for 
                 chronic conditions.

            Your job: Read the question and determine which ONE category above best describes the main focus of the question. 
            Return only the category name in your response. DO NOT give reasoning or extra text.
            """},
            {"role": "user", "content": f"""
            Question:
            {message}

            Which ONE of these categories does it best fit?
            Answer ONLY with the category name (e.g., 'Diagnosis', 'Pharmacotherapy, Interventions and Management', etc.).
            No extra words, no explanations.
            """}
        ]
    )
    return response.choices[0].message.content.strip()

#input file path
with open('', 'r', encoding='utf-8') as file:
    data = json.load(file)  

for entry in tqdm(data, desc="Classifying", unit=" questions"):
    entry['category'] = query_openai_for_classification(entry['question'])  # Add classification

#output file path
with open('', 'w', encoding='utf-8') as outfile:
    json.dump(data, outfile, indent=4)