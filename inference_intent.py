import json
import tensorflow as tf
import numpy as np
import pickle

MODEL_DIR = "models"
DATA_PATH = "data/intents.json"

# Tải mô hình
model = tf.keras.models.load_model(f"{MODEL_DIR}/intent_model.keras")

# Tải vectorizer
with open(f"{MODEL_DIR}/vectorizer.json", "r", encoding="utf-8") as f:
    vec_conf = json.load(f)

vectorizer = tf.keras.layers.TextVectorization(
    output_mode="tf_idf",
    max_tokens=len(vec_conf["vocab"])
)
vectorizer.set_vocabulary(vec_conf["vocab"])

# Tải encoder
with open(f"{MODEL_DIR}/tag2id.pkl", "rb") as f:
    encoder = pickle.load(f)

# Tải dữ liệu intents
with open(DATA_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

# Hàm dự đoán
def predict_intent(text):
    X = vectorizer([text])
    pred = model.predict(X)
    tag = encoder.inverse_transform([np.argmax(pred)])[0]
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])
    return "Xin lỗi, mình chưa hiểu ý bạn 😅"

if __name__ == "__main__":
    while True:
        inp = input("Bạn: ")
        if inp.lower() in ["thoát", "exit", "quit"]:
            break
        print("Bot:", predict_intent(inp))
