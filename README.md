在微调、评估之前已经使用bitsandbytes==0.37.2, peft==0.3.0, transformers==4.28.0
## evaluate2.py
使用之前的评估文件，获得的auc  "64": 0.4636760882126348  
## evaluate.py 
使用这个评估文件，会出现此问题
```python
Loading checkpoint shards: 100%|██████████| 33/33 [00:08<00:00,  3.80it/s]
0it [00:00, ?it/s]Asking to truncate to max_length but no maximum length is provided and the model has no predefined maximum length. Default to no truncation.
32it [1:40:02, 187.58s/it]
32it [00:00, ?it/s]
Traceback (most recent call last):
  File "D:\mhf\TALLRec\evaluate.py", line 237, in <module>
    fire.Fire(main)
  File "D:\Anaconda\envs\alpaca\lib\site-packages\fire\core.py", line 141, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "D:\Anaconda\envs\alpaca\lib\site-packages\fire\core.py", line 475, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
  File "D:\Anaconda\envs\alpaca\lib\site-packages\fire\core.py", line 691, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "D:\mhf\TALLRec\evaluate.py", line 204, in main
    test_data[i]['logits'] = logits[i]
IndexError: list index out of range
```
## finetune_rec.py  
(1)去掉下面部分  
```python
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))
```
(2)发现如下：
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
(3)运行finetune_rec.py
获得lora-alpaca_movie_64_3文件夹
