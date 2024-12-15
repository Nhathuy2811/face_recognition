import os
import face_recognition
import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")

        # Biến kiểm soát webcam và nhận diện
        self.video_source = 0
        self.capture = None
        self.known_face_encodings = []  # Mảng lưu trữ các mã hóa khuôn mặt đã biết
        self.known_face_names = []  # Mảng lưu trữ tên của các khuôn mặt đã biết

        # Giao diện
        self.video_panel = tk.Label(root)
        self.video_panel.pack()

        self.start_button = tk.Button(root, text="Start Webcam", width=20, command=self.start_webcam)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(root, text="Stop Webcam", width=20, state=tk.DISABLED, command=self.stop_webcam)
        self.stop_button.pack(pady=10)

        self.load_button = tk.Button(root, text="Load Known Faces", width=20, command=self.load_known_faces)
        self.load_button.pack(pady=10)

        self.names_label = tk.Label(root, text="Known Faces: None", width=20)
        self.names_label.pack(pady=10)

    def load_known_faces(self):
        """Tải và mã hóa các khuôn mặt đã biết từ dataset"""
        dataset_dir = "dataset/"  # Đường dẫn đến dataset chứa ảnh

        # Lặp qua tất cả các thư mục trong dataset
        for person_name in os.listdir(dataset_dir):
            person_dir = os.path.join(dataset_dir, person_name)
            if os.path.isdir(person_dir):  # Chỉ xử lý nếu là thư mục
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)

                    # Mã hóa khuôn mặt từ ảnh
                    image = face_recognition.load_image_file(image_path)
                    face_encoding = face_recognition.face_encodings(image)

                    if face_encoding:  # Nếu phát hiện khuôn mặt trong ảnh
                        self.known_face_encodings.append(face_encoding[0])
                        self.known_face_names.append(person_name)

        # Cập nhật giao diện với các khuôn mặt đã biết
        self.names_label.config(text="Known Faces: " + ", ".join(self.known_face_names))
        messagebox.showinfo("Info", "Known faces loaded successfully")

    def start_webcam(self):
        """Khởi động webcam và bắt đầu nhận diện khuôn mặt"""
        self.capture = cv2.VideoCapture(self.video_source)
        self.stop_button.config(state=tk.NORMAL)
        self.update_frame()

    def stop_webcam(self):
        """Dừng webcam"""
        self.capture.release()
        cv2.destroyAllWindows()
        self.stop_button.config(state=tk.DISABLED)

    def update_frame(self):
        """Cập nhật khung hình video và nhận diện khuôn mặt"""
        ret, frame = self.capture.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Tìm vị trí khuôn mặt trong khung hình
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_names = []

            for encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, encoding)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                
                face_names.append(name)

            # Vẽ khung bao quanh khuôn mặt và tên
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            # Chuyển đổi khung hình OpenCV sang Tkinter có thể hiển thị
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            self.video_panel.imgtk = img
            self.video_panel.config(image=img)

            # Tiếp tục cập nhật khung hình mỗi 10ms
            self.root.after(10, self.update_frame)

# Khởi tạo giao diện Tkinter
root = tk.Tk()
app = FaceRecognitionApp(root)
root.mainloop()

