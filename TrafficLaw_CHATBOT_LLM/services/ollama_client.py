import requests
import json

class OllamaClient:
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/chat"

    def ask(self, system_prompt, user_message):
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False
        }

        try:
            response = requests.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.ConnectionError:
            return "⚠️ Không thể kết nối tới Ollama. Hãy chắc chắn rằng Ollama đang chạy (kiểm tra system tray hoặc chạy 'ollama serve')."
        except Exception as e:
            return f"⚠️ Có lỗi xảy ra: {str(e)}"