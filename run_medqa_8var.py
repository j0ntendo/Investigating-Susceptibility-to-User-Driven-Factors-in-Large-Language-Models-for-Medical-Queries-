from medqa.deepseek_8var import run_evaluation as deepseek_eval
from medqa.gemini_8var import run_evaluation as gemini_eval
from medqa.geminipro_8var import run_evaluation as geminipro_eval
from medqa.gpt_8var import run_evaluation as gpt_eval
from medqa.haiku_8var import run_evaluation as haiku_eval
from medqa.llama3_8var import run_evaluation as llama3_eval
from medqa.llama3med_8var import run_evaluation as llama3med_eval
from medqa.sonnet_8var import run_evaluation as sonnet_eval

DATA_DIR = ""  # data path like medqa_test.json

if __name__ == "__main__":
    deepseek_eval(DATA_DIR, "medqa_results/deepseek/")
    gemini_eval(DATA_DIR, "medqa_results/gemini/")
    geminipro_eval(DATA_DIR, "medqa_results/geminipro/")
    gpt_eval(DATA_DIR, "medqa_results/gpt/")
    haiku_eval(DATA_DIR, "medqa_results/haiku/")
    llama3_eval(DATA_DIR, "medqa_results/llama3/")
    llama3med_eval(DATA_DIR, "medqa_results/llama3med/")
    sonnet_eval(DATA_DIR, "medqa_results/sonnet/")