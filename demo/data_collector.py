import cv2 
import numpy as np

from tkinter import Tk, Canvas, BOTH
import keyboard
import os
import pickle
from monitor import monitor
import time
from datetime import datetime

mon = monitor()

DIR_TO_SAVE = "data"
new_dir_name = input("Input name dir:")
dir_to_save = os.path.join(DIR_TO_SAVE, new_dir_name)

if not os.path.exists(dir_to_save):
    os.makedirs(dir_to_save)
    print(f"Dir {dir_to_save} successfully created.")
else:
    print(f"Dir {dir_to_save} already exist.")


def print_circle(canvas, coords, r=100):
    c_x, c_y = coords
    line1 = canvas.create_line(c_x, c_y + r, c_x, c_y - r)
    line2 = canvas.create_line(c_x+r, c_y, c_x-r, c_y)
    oval = canvas.create_oval(c_x - r, c_y + r, c_x + r, c_y - r)

    return oval, line1, line2


root = Tk()

canvas = Canvas(root)
canvas.pack(fill=BOTH, expand=True)
canvas.focus_set()
# scr_w = canvas.winfo_screenwidth()
# scr_h = canvas.winfo_screenheight()

scr_w = mon.w_pixels
scr_h = mon.h_pixels

webcam = cv2.VideoCapture(0)

DIR_TO_SAVE = "data"

points = []
frames = []

def get_rand_point():
    x = int(np.random.uniform(0, 1) * scr_w)
    y = int(np.random.uniform(0, 1) * scr_h)
    return x,y
point = get_rand_point()
subjects = print_circle(canvas, point)
canvas.update()

i = 0
while True:
    _, frame = webcam.read()

    if keyboard.is_pressed(" "):
        frame_path = os.path.join(DIR_TO_SAVE, f"img_{i}.jpg")
        i+=1
        x_cam, y_cam, z_cam = mon.monitor_to_camera(*point)

        points.append((x_cam, y_cam))
        frames.append(frame)

        canvas.delete(*subjects)
        point = get_rand_point()
        subjects = print_circle(canvas, point)
        time.sleep(0.3)

    canvas.update()

    if keyboard.is_pressed("q"):
        time.sleep(0.3)
        canvas.destroy()
        break

    

for idx, img in enumerate(frames):
    frame_path = os.path.join(dir_to_save, f"img_{idx}.jpg")
    cv2.imwrite(frame_path, img)


points_path = os.path.join(dir_to_save, "points.pickle")

with open(points_path, 'wb') as file:
    pickle.dump(points, file)
