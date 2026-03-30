from transformers import AutoProcessor, AutoModelForImageTextToText
import json
from PIL import Image
import torch
import re

processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B")
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3.5-4B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

rgb_image = Image.open('test.jpg')

messages = [
    {
        "role": "system",
        "content":[
            {"type": "text", "text": """ 
You are a robot vision system.
Directly provide the output in JSON format without long explanations.
You will be given a user command and an image, from the command you should extract the users intent,
and from the image based on users intent you should find the target_object.
extract:
1. intent
2. target_object

Return Only JSON, the format should always be the same as below:
{
    "intent": "...",
    "target_object": "..."
}

"""}
        ]
    },
    {
        "role": "user",
        "content":[ 
            {"type": "image", "image":rgb_image},
            {"type":"text", "text": "목 말라!"}
        ]
    }
]
prompt = processor.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=False
)
inputs = processor(
	text = [prompt],
    images = [rgb_image],
	padding = True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=128, temperature =0.2)
response = processor.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)
print("RAW",response)
try:
    # 1. 정규표현식으로 가장 바깥쪽의 { } 내용을 찾습니다.
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        clean_json = json_match.group()
        data = json.loads(clean_json)
        
        # 2. 결과 출력
        intent = data.get("intent", "N/A")
        target = data.get("target_object", "N/A")
        
        print("-" * 30)
        print(f"✅ 의도 파악: {intent}")
        print(f"✅ 타겟 물체: {target}")
        print("-" * 30)
    else:
        print("❌ 에러: 결과에서 JSON 형태를 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ JSON 파싱 에러: {e}")
    print("모델의 실제 답변:", response)