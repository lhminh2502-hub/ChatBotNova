import json
import random
import streamlit as st
import tensorflow as tf
import numpy as np
import base64
import os
import time
from chatbot_gpt import call_gpt_api  # ⚙️ import GPT fallback

# =====================
# 🔹 Đọc file intents.json
# =====================
DATA_PATH = os.path.join("data", "intents.json")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

# =====================
# 🔹 Load mô hình và dữ liệu đã huấn luyện
# =====================
MODEL_PATH = os.path.join("models", "intent_model.keras")
CLASSES_PATH = os.path.join("models", "classes.json")
VECTORIZER_PATH = os.path.join("models", "vectorizer.json")
CONTEXT_FILE = "memory_context.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

with open(VECTORIZER_PATH, "r", encoding="utf-8") as f:
    vec_conf = json.load(f)

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=vec_conf.get("max_tokens", 1000),
    output_mode="count"
)
vectorizer.set_vocabulary(vec_conf["vocab"])
vectorizer.build(input_shape=(1,))

# =====================
# 🔹 HÀM CHÍNH
# =====================
def predict_intent(text):
    X = vectorizer([text])
    preds = model.predict(X)
    idx = np.argmax(preds)
    confidence = float(np.max(preds))  # lấy độ tin cậy
    return classes[idx], confidence

def get_response(intent):
    for item in intents["intents"]:
        if item["tag"] == intent:
            return random.choice(item["responses"])
    return ""

def load_context():
    if os.path.exists(CONTEXT_FILE):
        with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"context": []}

def save_context(data):
    with open(CONTEXT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def clear_context():
    save_context({"context": []})

# =====================
# 🔹 GIAO DIỆN STREAMLIT
# =====================
st.set_page_config(page_title="Chatbot Nova", page_icon="🌠", layout="centered")
# =====================
# 🔹 GIAO DIỆN STREAMLIT
# =====================
st.set_page_config(page_title="Chatbot Nova", page_icon="🌠", layout="centered")

# =========================
# 🔹 HÀM HỖ TRỢ: ĐỌC ẢNH LOCAL
# =========================
def load_local_image(image_path):
    """Chuyển ảnh local thành base64 để hiển thị trong HTML."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# =========================
# 🔹 SIDEBAR LIÊN HỆ NHÓM
# =========================
with st.sidebar:
    st.markdown("<h2 style='color:#f7b731;'>🤝 Liên hệ với đội ngũ Nova</h2>", unsafe_allow_html=True)

    # CSS cho thẻ thành viên
    st.markdown("""
    <style>
    .member-card {
        background-color: #2b2b2b;
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 12px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        transition: all 0.25s ease;
    }
    .member-card:hover {
        transform: scale(1.03);
        background-color: #333;
    }
    .member-img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        object-fit: cover;
        border: 2px solid orange;
        margin-bottom: 10px;
    }
    .member-name {
        font-weight: bold;
        color: #f7b731;
        margin-bottom: 4px;
    }
    .member-role, .member-sdt, .member-mail {
        font-size: 13px;
        color: #ccc;
        margin-bottom: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ======= ẢNH LOCAL =======
    img1 = load_local_image("images/hminh.png")
    img2 = load_local_image("images/Ctrang.jpg")  # đổi từ .ipg → .jpg (kiểm tra lại)
    img3 = load_local_image("images/gphuc.jpg")
    img4 = load_local_image("images/tloi.jpg")
    
    # ======= DANH SÁCH THÀNH VIÊN =======
    st.markdown(f"""
    <div class="member-card">
        <img src="data:image/png;base64,{img1}" class="member-img">
        <div class="member-name">Lê Hồng Minh</div>
        <div class="member-role">MSSV: 25800601139</div>
        <div class="member-sdt">SDT: 0787818993</div>
        <div class="member-mail">lhminh2502@gmail.com</div>
    </div>

    <div class="member-card">
        <img src="data:image/png;base64,{img2}" class="member-img">
        <div class="member-name">Nguyễn Hồng Công Trạng</div>
        <div class="member-role">MSSV: 25800600700</div>
        <div class="member-sdt">SDT: 0939119063</div>
        <div class="member-mail">congtrang2704@gmail.com</div>
    </div>

    <div class="member-card">
        <img src="data:image/png;base64,{img3}" class="member-img">
        <div class="member-name">Trần Gia Phúc</div>
        <div class="member-role">MSSV: 25800600542</div>
        <div class="member-sdt">SDT: 0704975731</div>
        <div class="member-mail">trangiaphuc1109pppp@gmail.com</div>
    </div>

    <div class="member-card">
        <img src="data:image/png;base64,{img4}" class="member-img">
        <div class="member-name">Văn Thanh Lợi</div>
        <div class="member-role">MSSV: 25800601215</div>
        <div class="member-sdt">SDT: 0362544602</div>
        <div class="member-mail">thanhloi29092007@gmail.com</div>
    </div>
    """, unsafe_allow_html=True)
# ===== CSS =====
st.markdown("""
<style>di
body { background-color: #181818; color: #f5f5f5; }
.chat-avatar {
    width: 38px; height: 38px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; margin: 6px;
}
.chat-line {
    display: flex; align-items: flex-start;
    margin-bottom: 14px;
    animation: fadeIn 0.25s ease, slideIn 0.25s ease;
}
.bot { justify-content: flex-start; }
.bot .chat-avatar { background-color: #f7b731; color: #000; }
.bot .chat-bubble {
    background-color: #2f2f2f; color: #f5f5f5;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px; max-width: 70%;
}
.user { justify-content: flex-end; }
.user .chat-avatar { background-color: #ff453a; color: #fff; }
.user .chat-bubble {
    background-color: #2b2b2b; color: #f5f5f5;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px; max-width: 70%; text-align: left;
}
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
</style>
""", unsafe_allow_html=True)

# =====================
# 🔹 LOGIC HIỂN THỊ
# =====================
st.title("🌠 Chatbot Nova")
st.caption("✨ Chatbot Nova do chúng tôi tạo.")

context_data = load_context()

# Nút xóa hội thoại
if st.button("🧹 Xóa hội thoại"):
    clear_context()
    st.rerun()

# Hiển thị hội thoại cũ
for chat in context_data["context"]:
    role_class = "user" if chat["role"] == "user" else "bot"
    avatar = "😊" if chat["role"] == "user" else "🤖"
    st.markdown(
        f"<div class='chat-line {role_class}'><div class='chat-avatar'>{avatar}</div><div class='chat-bubble'>{chat['text']}</div></div>",
        unsafe_allow_html=True
    )

# Ô nhập liệu
user_input = st.chat_input("Nhập tin nhắn của bạn...")

if user_input:
    # 🔸 Gộp ngữ cảnh 3 câu gần nhất
    recent_context = " ".join([m["text"] for m in context_data["context"][-3:]])
    combined_input = f"{recent_context} {user_input}"

    # 🔹 Dự đoán intent + độ tin cậy
    intent, confidence = predict_intent(combined_input)
    response = get_response(intent)
    
    # 🔹 Gọi GPT thật sự thay vì predict_intent
    from chatbot_gpt import call_gpt_api
    response = call_gpt_api(user_input)

    # Lưu hội thoại
    context_data["context"].append({"role": "user", "text": user_input})
    context_data["context"].append({"role": "bot", "text": response})
    save_context(context_data)

    # Hiển thị tin nhắn
    st.markdown(f"""
    <div class='chat-line user'>
        <div class='chat-bubble'>{user_input}</div>
        <div class='chat-avatar'>😊</div>
    </div>
    """, unsafe_allow_html=True)

    # Hiệu ứng bot đang gõ
    with st.chat_message("assistant"):
        thinking_msg = st.empty()
        for i in range(3):
            thinking_msg.markdown(f"💭 **Nova đang suy nghĩ{'.' * (i + 1)}**")
            time.sleep(0.5)

        typing_text = ""
        for ch in response:
            typing_text += ch
            thinking_msg.markdown(f"🪄 {typing_text}")
            time.sleep(0.02)

    st.rerun()
