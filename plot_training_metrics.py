import json
import matplotlib.pyplot as plt

# Загрузка логов
with open("training_logs.json", "r") as f:
    logs = json.load(f)

# Инициализация списков
train_steps = []
train_losses = []
eval_steps = []
eval_losses = []

# Заполняем списки
for log in logs:
    if "loss" in log:
        train_steps.append(log["step"])
        train_losses.append(log["loss"])
    if "eval_loss" in log:
        eval_steps.append(log["step"])
        eval_losses.append(log["eval_loss"])

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_losses, label="Train Loss", color='blue')
plt.plot(eval_steps, eval_losses, label="Eval Loss", color='red', marker='o')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid(True)
plt.savefig("training_loss_plot.png")
plt.show()
