from openai import OpenAI
from dotenv import load_dotenv
import os

# 🔹 Load biến môi trường
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🔹 Hàm gọi GPT
def call_gpt_api(user_message):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # hoặc gpt-4-turbo / gpt-4o
            messages=[
                {"role": "system", "content": "Bạn là Nova, một chatbot thân thiện và thông minh do Hminh tạo ra."},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Lỗi khi gọi GPT: {e}"
