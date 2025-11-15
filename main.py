import os
from platform import system

import requests
import speech_recognition as sr
from gtts import gTTS
import tempfile
import platform
import subprocess

from sympy.polys.polyconfig import query

GROQ_API_KEY = "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"
if not GROQ_API_KEY:
    raise RuntimeError("❌ Установіть GROQ_API_KEY (https://console.groq.com/keys)")

MODEL = "llama-3.3-70b-versatile"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def play_audio(path: str):
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.run(["mpg123", path])
    except Exception as e:
        print("Не вдалося відтворити відповідь!")

def listen_ukrainian(timeout=5, phrase_time_limit=20):
    r = sr.Recognizer()

    # 🎤 Тут створюється об’єкт мікрофона (можна змінити device_index)
    with sr.Microphone(device_index=1) as source:
        print("🎤 Говори... (українською)")
        print(sr.Microphone.list_microphone_names())
        r.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("⏱️ Не почув нічого.")
            return None
    try:
        text = r.recognize_google(audio, language="uk-UA")
        print("👂 Ти сказав:", text)
        return text
    except Exception:
        print("⚠️ Не вдалося розпізнати.")
        return None

def ask_groq(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "Ти україномовний помічник!"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        # "max_token": 500,
        "stream": False
    }

    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        if res.status_code != 200:
            print(res.status_code, res.text)
            return "Сталась помилка!"

        data = res.json()
        return data["choices"][0]["message"]["content"].strip()

    except Exception as err:
        print("Помилка", err)
        return 'Сталась помилка!'

def speak_ua(text):
    if not text:
        return
    try:
        tts = gTTS(text=text, lang="uk")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name
        tts.save(path)
        play_audio(path)
        os.remove(path)
    except Exception as err:
        print("Не вдалось озвучити!", err)


def main():
   while True:
       query = listen_ukrainian()
       if not query:
           continue
       if query.lower().strip() in ["вийти", "завершити"]:
           speak_ua("До зустрічі!")
           break

       print("Думаю ...")
       answer = ask_groq(query)
       print(f"Відповідь: {answer}")
       speak_ua(answer)



if __name__ == "__main__":
    main()