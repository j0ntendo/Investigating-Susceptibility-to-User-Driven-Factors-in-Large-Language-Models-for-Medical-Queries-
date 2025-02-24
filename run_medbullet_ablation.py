from medqa.ablation_deepseek import process_datasets as deepseek_eval
from medqa.ablation_geminiflash import process_datasets as geminiflash_eval
from medqa.ablation_geminipro import process_datasets as geminipro_eval
from medqa.ablation_gpt import process_datasets as gpt_eval
from medqa.ablation_haiku import process_datasets as haiku_eval
from medqa.ablation_llama3 import process_datasets as llama3_eval
from medqa.ablation_llama3med import process_datasets as llama3med_eval
from medqa.ablation_sonnet import process_datasets as sonnet_eval

DATA_DIR = ""  # Replace with your folder path that contains the 7 files

if __name__ == "__main__":
    deepseek_eval(DATA_DIR, "medbullets_results/deepseek/ablation")
    geminiflash_eval(DATA_DIR, "medbullets_results/gemini/ablation")
    geminipro_eval(DATA_DIR, "medbullets_results/geminipro/ablation")
    gpt_eval(DATA_DIR, "medbullets_results/gpt/ablation")
    haiku_eval(DATA_DIR, "medbullets_results/haiku/ablation")
    llama3_eval(DATA_DIR, "medbullets_results/llama3/ablation")
    llama3med_eval(DATA_DIR, "medbullets_results/llama3med/ablation")
    sonnet_eval(DATA_DIR, "medbullets_results/sonnet/ablation")