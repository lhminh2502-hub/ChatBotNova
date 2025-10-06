import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Đường dẫn
DATA_PATH = "data/intents.json"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Đọc dữ liệu
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

texts, labels = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# 2. Mã hóa nhãn
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# 3. Vector hóa văn bản
vectorizer = tf.keras.layers.TextVectorization(output_mode="tf_idf", max_tokens=1000)
text_ds = tf.data.Dataset.from_tensor_slices(texts).batch(32)
vectorizer.adapt(text_ds)
X = vectorizer(texts)

# 4. Xây dựng mô hình
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(set(labels)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 5. Huấn luyện
model.fit(X, y, epochs=200, verbose=0)
print("✅ Huấn luyện hoàn tất!")

# 6. Lưu mô hình và metadata
model.save(os.path.join(MODEL_DIR, "intent_model.keras"))

# Lưu vectorizer
vec_conf = {
    "vocab": vectorizer.get_vocabulary()
}
with open(os.path.join(MODEL_DIR, "vectorizer.json"), "w", encoding="utf-8") as f:
    json.dump(vec_conf, f, ensure_ascii=False, indent=2)

# Lưu encoder
with open(os.path.join(MODEL_DIR, "classes.json"), "w", encoding="utf-8") as f:
    json.dump(encoder.classes_.tolist(), f, ensure_ascii=False, indent=2)

with open(os.path.join(MODEL_DIR, "tag2id.pkl"), "wb") as f:
    pickle.dump(encoder, f)

print("💾 Đã lưu mô hình và dữ liệu thành công!")
