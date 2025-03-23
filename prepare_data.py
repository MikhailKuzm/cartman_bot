import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Загружаем модель перевода
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M",
    src_lang="eng_Latn",
    tgt_lang="rus_Cyrl",
    device=0  # -1 если без GPU
)

# Перевод строки
def translate_line(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    return translator(text.strip(), max_length=512)[0]["translation_text"]

# Загрузка CSV
df = pd.read_csv("All-seasons.csv")
df = df[['Season', 'Episode', 'Character', 'Line']]
df['Character'] = df['Character'].astype(str).str.lower()

# Контейнер для финальных примеров
dialogue_examples = []
window_size = 5

# Группируем по эпизодам
for (season, episode), group in tqdm(df.groupby(['Season', 'Episode'])):
    group = group.reset_index(drop=True)
    speakers = group['Character'].tolist()
    lines = group['Line'].tolist()

    for i in range(len(group)):
        if speakers[i] != 'cartman':
            continue  # только реплики Картмана — ответы

        # Формируем контекст
        context_start = max(0, i - window_size)
        context_lines = []
        for j in range(context_start, i):
            translated = translate_line(lines[j])
            token = "[OTHER]"
            if speakers[j] == 'cartman':
                token = "[CARTMAN]"
            context_lines.append(f"{token} {translated} {token.replace('[', '[/')}")

        if not context_lines:
            continue

        # Переводим ответ Картмана
        response_translated = translate_line(lines[i])
        response_tagged = f"[CARTMAN] {response_translated} [/CARTMAN]"

        dialogue_examples.append({
            "context_ru": "\n".join(context_lines).strip(),
            "response_ru": response_tagged,
            "season": season,
            "episode": episode
        })

# Сохраняем
result_df = pd.DataFrame(dialogue_examples)
result_df.to_csv('cartman_dialogue_dataset_ru_safe')

result_df = pd.read_csv('cartman_dialogue_dataset_ru_safe.csv')
# Фильтрация плохих примеров
def is_valid(text):
    words = text.strip().split()
    if len(words) < 3:
        return False
    for w in set(words):
        if words.count(w) > len(words) * 0.5 and len(words) > 5:
            return False
    for letter in text:
        if text.count(letter) > len(text) * 0.5:
            return False
    return True

df_filtered = result_df[ result_df["context_ru"].apply(is_valid) & result_df["response_ru"].apply(is_valid)]
tmp1 = pd.read_csv('cartman_real_verified_20_ru.csv')
tmp2 = pd.read_csv('cartman_real_verified_15_ru_manual.csv')

df_filtered = pd.concat([df_filtered, tmp1, tmp2])

# Разделение на train/test (95% / 5%)
train_df, test_df = train_test_split(df_filtered, test_size=0.05, random_state=42)

# Сохранение
train_path = "cartman_train_clean.csv"
test_path = "cartman_test_clean.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)