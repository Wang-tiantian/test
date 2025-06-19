print("##################start#############")
import os
os.environ["VLLM_LOCAL_IP"] = "127.0.0.1"  # 绕过IP探测
os.environ["HF_HUB_OFFLINE"] = "1"         # 强制离线模式
import re
import json
import torch
import pandas as pd
from pandas import json_normalize
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from peft import LoraConfig, PeftModel
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

print("##################data#############")
test_file = "data/data_for_llms/thermal_test.csv"
test_df = pd.read_csv(test_file, encoding='utf-8')
print(test_df.info())

instruction = '''Extract the reactionDetails from the articleInformation and mainReactions.'''
test_df['text'] = f'[INST] {instruction} ' + test_df["articleInformation"] + " [/INST] " +  test_df["mainReactions"] + " [/INST] "
prompts = list(test_df['text'])

print("##################model#############")

new_model_name = "saved_models/self_Qwen2.5-14B-Instruct/train_231_lr5e-06_bs2/checkpoint-2088"
sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens = 4096, stop = ['!!!'])
llm = LLM(model = new_model_name, tensor_parallel_size=1, trust_remote_code=True)

print("##################generate#############")

outputs = llm.generate(prompts, sampling_params)
predictions = []

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt},\nGenerated text: {generated_text!r}")
    predictions.append(generated_text.strip())

pred_df = pd.DataFrame()
pred_df['Generated Text'] = predictions
pred_df['Actual Text'] = test_df["reactionDetails"]
pred_df['mainReactions'] = test_df["mainReactions"]
pred_df['articleInformation'] = test_df["articleInformation"]
pred_df.to_csv(f"results/predictions/prediction_of_{new_model_name.replace('/', '-')}.csv", index = None)
pred_df