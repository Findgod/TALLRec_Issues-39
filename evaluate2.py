import sys

import fire
# import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

# 定义主函数 main，用于评估模型性能
def main(
    load_8bit: bool = False,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "lora-alpaca_movie_64_3",
    test_data_path: str = "data/movie/test.json",
    result_json_data: str = "temp.json",
    share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    train_sce = 'movie'
    test_sce = ' movie'
    model_name = 'lora-alpaca_movie_64_3'
    seed = 3
    sample = 64

    # # 解析 lora_weights 以获得模型类型、模型名称、训练和测试场景、种子和样本
    # model_type = lora_weights.split('/')[-1]
    # model_name = '_'.join(model_type.split('_')[:2])
    #
    # if model_type.find('book') > -1:
    #     train_sce = 'book'
    # else:
    #     train_sce = 'movie'
    #
    # if test_data_path.find('book') > -1:
    #     test_sce = 'book'
    # else:
    #     test_sce = 'movie'
    #
    #
    #
    # temp_list = model_type.split('_')
    # print(temp_list)
    # seed = temp_list[-2]
    # sample = temp_list[-1]




    # 从 result_json_data 文件中加载数据
    if not os.path.exists(result_json_data):
        data = {}
        json_data = json.dumps(data)
        with open(result_json_data, 'w') as f:
            f.write(json_data)
    f = open(result_json_data, 'r')
    data = json.load(f)
    f.close()

    # 初始化数据结构，以存储结果
    if not data.__contains__(train_sce):
        data[train_sce] = {}
    if not data[train_sce].__contains__(test_sce):
        data[train_sce][test_sce] = {}
    if not data[train_sce][test_sce].__contains__(model_name):
        data[train_sce][test_sce][model_name] = {}
    if not data[train_sce][test_sce][model_name].__contains__(seed):
        data[train_sce][test_sce][model_name][seed] = {}
    if data[train_sce][test_sce][model_name][seed].__contains__(sample):
        exit(0)
        # data[train_sce][test_sce][model_name][seed][sample] = 

    # 使用 LlamaTokenizer 加载 tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    # 根据不同的设备设置，加载 LlamaForCausalLM 模型和 LOLA 权重
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # # 解决 decapoda-research 配置中的问题
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # 定义 evaluate 函数，用于执行生成操作并评估模型性能
    def evaluate(
        instruction,
        input=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        **kwargs,
    ):  # 生成输入的提示文本
        prompt = generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        L = input_ids.shape[1]
        # print(L)
        with torch.no_grad():
            # 执行生成操作
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        # 获取生成的分数和 logits
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = torch.tensor([scores[0][8241], scores[0][3782]], dtype=torch.float32).softmax(dim=-1)
        s = generation_output.sequences[0]
        output = tokenizer.decode(s[L:], skip_special_tokens=True)
        return output.strip(' '), logits

    # testing code for readme

    # 用于测试的代码示例
    logit_list = []
    gold_list= []
    from tqdm import tqdm
    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        for i, test in tqdm(enumerate(test_data)):
            if test['output'] == 'Yes.':
                gold_list.append(1)
            else:
                gold_list.append(0)
            output, logits = evaluate(test['instruction'], test['input'])
            logit_list.append(logits[0].item())
    
    # print(roc_auc_score(gold_list, logit_list))
    # 计算 ROC AUC 并将结果存储到数据中
    data[train_sce][test_sce][model_name][seed][sample] = roc_auc_score(gold_list, logit_list)
    f = open(result_json_data, 'w')
    json.dump(data, f, indent=4)
    f.close()

# 生成模型输入的提示文本
def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)
