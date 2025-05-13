import os
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2


def get_monitor_params(calib_dir):
    monitor = scipy.io.loadmat(os.path.join(calib_dir, "monitorPose.mat"))
    screen = scipy.io.loadmat(os.path.join(calib_dir, "screenSize.mat"))

    R_monitor, _ = cv2.Rodrigues(monitor["rvects"])
    T_monitor = monitor["tvecs"].reshape(3)

    screen_mm = {
        "w": screen["width_mm"][0][0],
        "h": screen["height_mm"][0][0],
        "w_px": screen["width_pixel"][0][0],
        "h_px": screen["height_pixel"][0][0],
    }

    return R_monitor, T_monitor, screen_mm


def intersect_ray_plane(eye, ray, plane_point, plane_normal):
    denom = np.dot(ray, plane_normal)
    if np.abs(denom) < 1e-6:
        return None
    t = np.dot(plane_point - eye, plane_normal) / denom
    return eye + t * ray


def gaze_to_screen_xy(eye_pos, gaze_dir, R, T, screen_mm):
    plane_normal = R[:, 2]
    plane_point = T
    hit = intersect_ray_plane(eye_pos, gaze_dir, plane_point, plane_normal)
    if hit is None:
        return None

    screen_coords_mm = R.T @ (hit - T)

    x_mm = screen_coords_mm[0]
    y_mm = screen_coords_mm[1]

    x_norm = x_mm / screen_mm["w"]
    y_norm = y_mm / screen_mm["h"]
    return torch.tensor([x_norm, y_norm], dtype=torch.float32)


class MPIIGazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for subject in sorted(os.listdir(root_dir)):
            subj_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subj_path) or not subject.startswith("p"):
                continue

            calib_dir = os.path.join(subj_path, "Calibration")
            R, T, screen_mm = get_monitor_params(calib_dir)

            for day in sorted(os.listdir(subj_path)):
                day_path = os.path.join(subj_path, day)
                if not os.path.isdir(day_path) or not day.startswith("day"):
                    continue

                anno_file = os.path.join(day_path, "annotation.txt")
                if not os.path.exists(anno_file):
                    continue

                with open(anno_file, "r") as f:
                    lines = f.readlines()

                for idx, line in enumerate(lines):
                    values = list(map(float, line.strip().split()))
                    if len(values) < 41:
                        continue

                    image_path = os.path.join(day_path, f"{idx + 1:04d}.jpg")
                    if not os.path.exists(image_path):
                        raise Exception

                    eye = np.array(values[35:38])
                    gaze = np.array(values[26:29])
                    gaze = gaze / np.linalg.norm(gaze)

                    screen_xy = gaze_to_screen_xy(eye, gaze, R, T, screen_mm)
                    if screen_xy is None:
                        raise Exception

                    self.samples.append((image_path, screen_xy))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, screen_xy = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, screen_xy
