import sys
import cv2
import torch
import numpy as np
import random
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor
from torchvision import transforms
from PIL import Image
from models.Haar import find_eyes
from models.GazeCNN import GazeCNN

model = GazeCNN()
dict = torch.load('GazeCNN.pth', map_location=torch.device('cpu'))
cleaned = {k.replace('_orig_mod.', ''): v for k, v in dict.items()}
model.load_state_dict(cleaned)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class GazeApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gaze Estimation")
        self.showFullScreen()

        self.screen_width = QApplication.primaryScreen().size().width()
        self.screen_height = QApplication.primaryScreen().size().height()

        self.dot_pos = self.random_dot_position()
        self.best_prediction = None
        self.best_error = float('inf')

        self.screen_label = QLabel()
        self.screen_label.setFixedSize(self.screen_width, self.screen_height - 200)
        self.screen_label.setStyleSheet("background-color: white;")

        self.cam_label = QLabel("Camera Preview")
        self.cam_label.setFixedSize(320, 240)

        self.result_label = QLabel("Prediction: N/A")
        self.change_dot_btn = QPushButton("Change Dot Position")
        self.change_dot_btn.clicked.connect(self.change_dot_position)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.screen_label)

        h_layout = QHBoxLayout()
        h_layout.addWidget(self.cam_label)
        h_layout.addWidget(self.result_label)
        h_layout.addWidget(self.change_dot_btn)
        h_layout.addWidget(self.exit_btn)

        layout.addLayout(h_layout)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)  # Snapshot every second

    def random_dot_position(self):
        x = random.randint(50, self.screen_width - 50)
        y = random.randint(50, self.screen_height - 250)
        self.best_prediction = None
        self.best_error = float('inf')
        return QPoint(x, y)

    def change_dot_position(self):
        self.dot_pos = self.random_dot_position()
        self.repaint()

    def paint_screen(self, predicted_pos=None):
        pixmap = QPixmap(self.screen_label.size())
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        painter.setBrush(Qt.red)
        painter.drawEllipse(self.dot_pos, 10, 10)  # Ground truth

        if predicted_pos:
            painter.setBrush(Qt.blue)
            painter.drawEllipse(predicted_pos, 10, 10)

        if self.best_prediction:
            painter.setBrush(Qt.green)
            painter.drawEllipse(self.best_prediction, 10, 10)

        painter.end()
        self.screen_label.setPixmap(pixmap)

    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Show camera feed
        cam_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = cam_rgb.shape
        cam_img = QImage(cam_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        cam_pixmap = QPixmap.fromImage(cam_img).scaled(self.cam_label.size(), Qt.KeepAspectRatio)
        self.cam_label.setPixmap(cam_pixmap)

        eye_img = find_eyes(frame)
        if eye_img is None:
            return

        pil_img = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor).squeeze().numpy()
            x_offset = 0.1 
            y_offset = 1.25
            output[0] = np.clip(output[0] + x_offset, 0, 1)
            output[1] = np.clip(output[1] ** y_offset, 0, 1)
            pred_x = int(output[0] * self.screen_label.width())
            pred_y = int(output[1] * self.screen_label.height())
            predicted_pos = QPoint(pred_x, pred_y)

            # Calculate error
            dx = self.dot_pos.x() - pred_x
            dy = self.dot_pos.y() - pred_y
            error = np.sqrt(dx ** 2 + dy ** 2)
            max_dist = np.sqrt(self.screen_width ** 2 + (self.screen_height - 200) ** 2)
            error_percent = (error / max_dist) * 100

            if error < self.best_error:
                self.best_error = error
                self.best_prediction = predicted_pos

            self.result_label.setText(
                f"Prediction: ({pred_x}, {pred_y})  Error: {error_percent:.1f}%\nBest: ({self.best_prediction.x()}, {self.best_prediction.y()})  Min Error: {self.best_error:.1f} px")

            self.paint_screen(predicted_pos)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = GazeApp()
    win.show()
    sys.exit(app.exec_())
