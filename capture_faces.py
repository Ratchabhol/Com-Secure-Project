import cv2
import os

# ตั้งชื่อ User ID สำหรับผู้ใช้งานใหม่
user_id = input("Enter User ID: ")  # เช่น "123"
output_dir = f"dataset/{user_id}"  # เก็บรูปในโฟลเดอร์ dataset/123

# สร้างโฟลเดอร์สำหรับเก็บข้อมูล (ถ้ายังไม่มี)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# เปิดกล้อง
cap = cv2.VideoCapture(1) # ใช้กล้องที่ 1 (0 สำหรับกล้องหลัก)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Press 'q' to stop capturing images.")
count = 0
50
while True:
    # อ่านเฟรมจากกล้อง
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้าในภาพ
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # เก็บใบหน้าที่ตรวจจับได้
    for (x, y, w, h) in faces:
        count += 1
        face = gray[y:y+h, x:x+w]
        file_path = os.path.join(output_dir, f"{count}.jpg")
        cv2.imwrite(file_path, face)  # บันทึกภาพใบหน้า

        # วาดกรอบรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # แสดงภาพพร้อมกรอบ
    cv2.imshow("Capturing Faces", frame)

    # หยุดเมื่อกด 'q' หรือเก็บครบ 50 รูป
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Saved {count} images in {output_dir}")
