from medbullet.deepseek_8var import run_evaluation as deepseek_eval
from medbullet.geminiflash_8var import run_evaluation as gemini_eval
from medbullet.geminipro_8var import run_evaluation as geminipro_eval
from medbullet.gpt_8var import run_evaluation as gpt_eval
from medbullet.haiku_8var import run_evaluation as haiku_eval
from medbullet.llama3_8var import run_evaluation as llama3_eval
from medbullet.llama3med_8var import run_evaluation as llama3med_eval
from medbullet.sonnet_8var import run_evaluation as sonnet_eval

DATA_DIR = ""  # data path like medbullet_test.json

if __name__ == "__main__":
    deepseek_eval(DATA_DIR, "medbullets_results/deepseek/")
    gemini_eval(DATA_DIR, "medbullets_results/gemini/")
    geminipro_eval(DATA_DIR, "medbullets_results/geminipro/")
    gpt_eval(DATA_DIR, "medbullets_results/gpt/")
    haiku_eval(DATA_DIR, "medbullets_results/haiku/")
    llama3_eval(DATA_DIR, "medbullets_results/llama3/")
    llama3med_eval(DATA_DIR, "medbullets_results/llama3med/")
    sonnet_eval(DATA_DIR, "medbullets_results/sonnet/")