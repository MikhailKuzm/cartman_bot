import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
import json

# ==== Логгер ====
class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.logs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs["step"] = state.global_step
            self.logs.append(logs)
            print(f"Step {state.global_step} |", " | ".join(f"{k}: {v:.4f}" for k, v in logs.items() if isinstance(v, float)))
            with open("training_logs.json", "w") as f:
                json.dump(self.logs, f, indent=2)

# ==== Подгрузка данных ====
train_df = pd.read_csv("cartman_train_clean.csv")
test_df = pd.read_csv("cartman_test_clean.csv")
train_df["text"] = train_df["context_ru"] + "\n" + train_df["response_ru"]
test_df["text"] = test_df["context_ru"] + "\n" + test_df["response_ru"]
train_dataset = Dataset.from_pandas(train_df[["text"]])
test_dataset = Dataset.from_pandas(test_df[["text"]])

# ==== Модель и токенизатор ====
model_name = "sberbank-ai/rugpt3small_based_on_gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# ==== Токенизация ====
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

train_tokenized = train_dataset.map(tokenize, batched=True)
test_tokenized = test_dataset.map(tokenize, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==== Аргументы обучения ====
training_args = TrainingArguments(
    output_dir="./cartman_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_strategy="steps",     # вывод логов по шагам
    logging_steps=10,             # каждые 10 шагов
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# ==== Trainer ====
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[LoggingCallback()]
)

trainer.train()
tokenizer.save_pretrained("cartman_tokinizer")
trainer.save_model("cartman_model_best")
