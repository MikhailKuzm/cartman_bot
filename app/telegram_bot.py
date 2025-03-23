import telebot
from app.inference import generate_cartman_reply
import random

# 🔐 Подставь свой Telegram токен
API_TOKEN = '###'
bot = telebot.TeleBot(API_TOKEN)

# 💬 История сообщений по каждому пользователю
user_histories = {}

# 🧱 Формируем контекст в формате [OTHER] ... [/OTHER] и [CARTMAN]
def build_context(user_id, max_turns=5):
    history = user_histories.get(user_id, [])
    # Только последние max_turns*2 фраз
    history = history[-(max_turns):]
    context = ""
    for turn in history:
        tag = "[OTHER]" if turn["role"] == "OTHER" else "[CARTMAN]"
        context += f"{tag} {turn['text']} {tag.replace('[', '[/')}\n"

    # 🔥 если в истории нет реплик Картмана — добавим шаблон
    if not any(turn["role"] == "CARTMAN" for turn in history):
        context += "[OTHER] Кто ты вообще такой? [/OTHER]\n[CARTMAN] Я — Картман, тупица! [/CARTMAN]\n"
    return context, history


# /start
@bot.message_handler(commands=["start"])
def handle_start(message):
    pass

# 📬 Обработка сообщений
@bot.message_handler(content_types=['text'])
def handle_message(message):
    user_id = message.chat.id
    user_text = message.text.strip()

    # Формируем контекст + новая реплика
    context, history = build_context(user_id)
    full_context = f"{context}[OTHER] {user_text} [/OTHER]\n[CARTMAN]"
    reply = generate_cartman_reply(full_context)
    # Обновляем историю
    history.append({"role": "OTHER", "text": user_text})
    history.append({"role": "CARTMAN", "text": reply})
    user_histories[user_id] = history

    bot.send_message(chat_id=user_id, text=reply)

# 🚀 Запуск бота
if __name__ == '__main__':
    print("🤖 Картман-бот запущен. Ожидаю сообщения...")
    bot.infinity_polling()
