import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Путь к обученной модели
TOKENIZER_PATH = "cartman_tokenizer"
MODEL_PATH = "cartman_model_best"

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.to('cpu')
model.eval() 
# Устанавливаем EOS и PAD токен
tokenizer.eos_token = '[/CARTMAN]'
tokenizer.pad_token = tokenizer.eos_token  
eos_token_id = tokenizer.convert_tokens_to_ids('[/CARTMAN]')

def generate_cartman_reply(context: str, max_new_tokens: int = 64) -> str:
    if not context.strip().endswith("[CARTMAN]"):
        context = context.strip() + "\n[CARTMAN] "
    
    inputs = tokenizer(context, return_tensors="pt", truncation=True) 

 
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=20,
            top_p=0.9,
            temperature=0.9,
            repetition_penalty=1.2,
            eos_token_id=eos_token_id,
            pad_token_id= eos_token_id 
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "[CARTMAN]" in decoded:
        reply = decoded.split("[CARTMAN]")[-1].replace('[/CARTMAN]', '')
        return reply.strip()

    return decoded.strip() 
