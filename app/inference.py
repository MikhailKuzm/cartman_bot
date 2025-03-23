import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Путь к обученной модели
MODEL_PATH = "app/cartman_model_best"

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(f'{MODEL_PATH}\checkpoint-565')
model.to('cpu')
model.eval()

def generate_cartman_reply(context: str, max_new_tokens: int = 64) -> str:
    if not context.strip().endswith("[CARTMAN]"):
        context = context.strip() + "\n[CARTMAN]"

    inputs = tokenizer(context, return_tensors="pt", truncation=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0])
    print("🧠 Generated raw text:\n", decoded)

    # Извлекаем первую реплику [CARTMAN] ... [/CARTMAN]
    start = decoded.find("[CARTMAN]")
    end = decoded.find("[/CARTMAN]", start)

    if start != -1 and end != -1:
        return decoded[start + len("[CARTMAN]"):end].strip()

    # fallback: взять всё после последнего [CARTMAN]
    fallback = decoded.split("[CARTMAN]")[-1]
    fallback = fallback.split("[OTHER]")[0]
    return fallback.strip()


# Пример запуска
if __name__ == "__main__":
    test_context = """
    [OTHER] Эй, Картман, ты опять проспал школу? [/OTHER]
    [OTHER] Ты вчера обещал прийти пораньше. [/OTHER]
    [CARTMAN]"""

    reply = generate_cartman_reply(test_context)
    print(f"[CARTMAN] {reply}")
