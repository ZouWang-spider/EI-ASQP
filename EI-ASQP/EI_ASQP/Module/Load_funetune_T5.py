import torch
from peft import PeftModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

#加载微调后T5模型参数
tokenizer = T5Tokenizer.from_pretrained("/Causal_ASQP\ModelPath\Laptop_T5_based")
base_model = T5ForConditionalGeneration.from_pretrained("D:\T5_Based")
model = PeftModel.from_pretrained(base_model, "/Causal_ASQP\ModelPath\Laptop_T5_based")



# 设置模型为 eval 模式
model.eval()

from Causal_ASQP.BaseModel.Prompt import construct_prompt
sentence = "Battery life is good but screen is faulty"
input_text = construct_prompt(sentence)

print(input_text)


# 编码输入文本
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# 使用模型生成结果
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=4
    )

# 解码生成的输出
decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 打印结果
print("模型输出：", decoded)