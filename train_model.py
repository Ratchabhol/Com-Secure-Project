import cv2
import numpy as np
import os

# เตรียมข้อมูลจากโฟลเดอร์ dataset
data_dir = "dataset"
recognizer = cv2.face.LBPHFaceRecognizer_create()  # สร้างโมเดลจดจำใบหน้า

faces = []  # เก็บภาพใบหน้า
ids = []    # เก็บ User ID

# อ่านข้อมูลจาก dataset
for user_id in os.listdir(data_dir):
    print(f"Processing User ID: {user_id}")  # Debug: แสดง User ID
    user_dir = os.path.join(data_dir, user_id)
    for image_file in os.listdir(user_dir):
        print(f"Loading image: {image_file}")  # Debug: แสดงชื่อไฟล์ภาพ
        img_path = os.path.join(user_dir, image_file)

        # โหลดภาพในรูปแบบขาวดำ (Grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # ปรับขนาดภาพให้เป็น 100x100
        img = cv2.resize(img, (100, 100))

        # เพิ่มภาพและ User ID ลงในลิสต์
        faces.append(img)
        ids.append(int(user_id))

# แปลงลิสต์ให้เป็น NumPy Array
faces = np.array(faces)
ids = np.array(ids)

# เทรนโมเดลจดจำใบหน้า
print("Training the model, please wait...")
recognizer.train(faces, ids)

# บันทึกโมเดลในไฟล์ face_trainer.yml
recognizer.save("face_trainer.yml")
print("Model trained and saved as 'face_trainer.yml'")