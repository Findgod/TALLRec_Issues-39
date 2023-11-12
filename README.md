## evaluate2.py
是之前的评估文件，auc  "64": 0.4636760882126348  
## evaluate.py 
只修改对应的路径等参数  
## finetune_rec.py  
### (1)去掉下面部分  
```python
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))
```
### (2)发现如下：
没有quantization_config时，64个样本训练时间：'train_runtime': 79.9546  
加上quantization_config时，64个样本训练时间：'train_runtime': 1319.9201  
```python
    # quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        # quantization_config=quantization_config,
    )
```
