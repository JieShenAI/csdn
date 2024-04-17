# from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
device = "cuda" # the device to load the model onto

model_name = r'C:\Users\jshen\.cache\modelscope\hub\qwen\Qwen1___5-1___8B-Chat-GPTQ-Int4'
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
).eval()
# model.generation_config = GenerationConfig.from_pretrained(
#                                 "qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",trust_remote_code=True)		

# model.generation_config.do_sample = False

# print(model.generation_config)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "你是谁？"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
# model_inputs = tokenizer([text], return_tensors="pt").to(device)

# generated_ids = model.generate(
#     model_inputs.input_ids,
#     max_new_tokens=512,
# )

# generated_ids = [
#     output_ids[len(input_ids):] 
#     for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
# ]

# response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(response)

response, history = model.chat(
        tokenizer,
        text,
        history=[["你好，我是小明，今年18岁了。", "你好，我是Qwen!"]],
        max_length=2048,  # 如果未提供最大长度，默认使用2048
        top_p=0.7,  # 如果未提供top_p参数，默认使用0.7
        temperature=0.95  # 如果未提供温度参数，默认使用0.95
    )

print(history)
print(response)