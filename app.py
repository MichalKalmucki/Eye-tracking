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

    
        self.calibrated = False
        self.calibration_points = [
            QPoint(50, 50),  # top-left
            QPoint(self.screen_label.width() - 50, 50),  # top-right
            QPoint(50, self.screen_label.height() - 50),  # bottom-left
            QPoint(self.screen_label.width() - 50, self.screen_label.height() - 50),  # bottom-right
            QPoint(self.screen_label.width() // 2, self.screen_label.height() // 2)  # center
        ]
        self.current_calib_index = 0
        self.calib_predictions = []
        self.calib_timer = QTimer()
        self.calib_timer.timeout.connect(self.collect_calib_data)
        self.calib_timer.start(1000)
        self.calib_dot_samples = []
        self.x_offset = 0.0
        self.y_offset = 0.0

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

    def collect_calib_data(self):
        if self.current_calib_index >= len(self.calibration_points):
            self.apply_calibration()
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        eye_img = find_eyes(frame)
        if eye_img is None:
            return

        pil_img = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor).squeeze().numpy()
            self.calib_dot_samples.append(output)

        # show calibration dot
        self.dot_pos = self.calibration_points[self.current_calib_index]
        self.paint_screen()

        if len(self.calib_dot_samples) >= 5:
            mean_pred = np.mean(self.calib_dot_samples, axis=0)
            self.calib_predictions.append(mean_pred)
            self.calib_dot_samples.clear()
            self.current_calib_index += 1

    def apply_calibration(self):
        pred_xs = [p[0] for p in self.calib_predictions]
        pred_ys = [p[1] for p in self.calib_predictions]
        true_xs = [p.x() / self.screen_label.width() for p in self.calibration_points]
        true_ys = [p.y() / self.screen_label.height() for p in self.calibration_points]

        pred_range_x = max(pred_xs) - min(pred_xs)
        pred_range_y = max(pred_ys) - min(pred_ys)
        true_range_x = max(true_xs) - min(true_xs)
        true_range_y = max(true_ys) - min(true_ys)

        self.x_scale = true_range_x / pred_range_x if pred_range_x != 0 else 1.0
        self.y_scale = true_range_y / pred_range_y if pred_range_y != 0 else 1.0

        self.x_shift = min(true_xs) - self.x_scale * min(pred_xs)
        self.y_shift = min(true_ys) - self.y_scale * min(pred_ys)

        self.calibrated = True
        self.calib_timer.stop()
        self.timer.start(1000)

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
            if not self.calibrated:
                return  

            output[0] = np.clip(output[0] * self.x_scale + self.x_shift, 0, 1)
            output[1] = np.clip(output[1] * self.y_scale + self.y_shift, 0, 1)

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
