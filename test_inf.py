import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = 'lmsys/vicuna-7b-v1.5'
LORA_PATH = './vicuna_final_lora_model'
DEVICE = 'cuda'

print('loading model...')
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config, device_map='auto')
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
VICUNA_TEMPLATE = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + ' ' }}{% elif message['role'] == 'user' %}{{ 'USER: ' + message['content'] + ' ' }}{% elif message['role'] == 'assistant' %}{{ 'ASSISTANT: ' + message['content'] + '</s>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:' }}{% endif %}"
tokenizer.chat_template = VICUNA_TEMPLATE

df = pd.read_excel('LoRA_testing_demo_ready.xlsx')

def eval_row(row, debug=False):
    chat = [
        {'role': 'system', 'content': 'You are a professional purchase behavior predictor and an expert in trust research.'},
        {'role': 'user', 'content': f"Trust is a multi-dimensional construct composed of Benevolence, Integrity, and Competence. Benevolence reflects goodwill and care toward customers (e.g., proactive communication and support). Integrity reflects honesty and transparency in business dealings. Competence reflects the ability and expertise to deliver expected service, often associated with experience and performance. Trust is a composite score derived from these three dimensions. Analyze the following values: Benevolence: {row['B']}, Integrity: {row['I_reverse']}, Competence: {row['C']}, Trust: {round(row['trust'], 4)}. Predict if this user will purchase again. Output '1' if the user will buy again, or '0' if they will not. Answer with ONLY the number (0 or 1)."}
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    if debug: print("\nDEBUG PROMPT:\n" + repr(prompt))

    inputs = tokenizer(prompt, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return pred_text

c0 = 0
c1 = 0
for i, row in df.iterrows():
    p = eval_row(row, debug=(i==0))
    p_int = 1 if '1' in p else 0
    if p_int == 1: c1+=1
    else: c0+=1
    if i < 3:
        print(f"[{i}] Predict: {repr(p)} Label: {row['再購']}")

print(f"Total: 1s={c1}, 0s={c0}")
