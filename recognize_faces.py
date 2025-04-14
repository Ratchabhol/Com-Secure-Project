import cv2

# โหลดโมเดลที่เทรนไว้
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# เปิดกล้อง
cap = cv2.VideoCapture(0)

print("Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))  # ปรับขนาดให้ตรงกับที่เทรน

        # ทำนาย User ID และ Confidence
        user_id, confidence = recognizer.predict(face)

        if confidence < 70:  # กำหนด threshold
            text = f"User {user_id} (Conf: {confidence:.2f})"
        else:
            text = "Unknown"

        # วาดกรอบใบหน้า
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # แสดงภาพที่ตรวจจับ
    cv2.imshow("Face Recognition", frame)

    # กด 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
