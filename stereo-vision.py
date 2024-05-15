"""
José Miguel González Zaragoza
631145 IRSI 6to
UDEM
Dr. Andrés Hernández Gutiérrez

stereo-vision
How to use:

$ python stereo-vision.py --l_img left_infrared_image.png --r_img right_infrared_image.png
"""

import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt


def read_parse_images()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_img', 
                        type=str, 
                        required=True,
                        help="Left Image is needed")
    parser.add_argument('--r_img',
                        type=str,
                        required=True,
                        help="Right Image is needed")
    args=parser.parse_args()
    return args

def display_images(left_image_path: str, right_image_path: str):
    left_img = cv.imread(left_image_path)
    right_img = cv.imread(right_image_path)
    
    # Check if images are loaded successfully
    if left_img is None or right_img is None:
        print("Error: Could not read the images.")
        return
    
    # Display images
    cv.imshow("Left Image", left_img)
    cv.imshow("Right Image", right_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def collect_coordinates(left_image_path: str, right_image_path: str):
    left_coords = []
    right_coords = []

    def on_mouse_click(event, x, y, flags, param):
        nonlocal left_coords, right_coords

        if event == cv.EVENT_LBUTTONDOWN:
            if param == 'left':
                if len(left_coords) >= 30:
                    draw_message("You have made your 30 clicks, proceed with the next 30 in the other image")
                else:
                    left_coords.append((x, y))
                    draw_message(f"Left image: {30 - len(left_coords)} clicks left")
            elif param == 'right':
                if len(left_coords) < 30:
                    draw_message("Please click 30 times on the left image first.")
                else:
                    if len(right_coords) >= 30:
                        draw_message("You have made your 30 clicks, proceed with the next steps.")
                    else:
                        right_coords.append((x, y))
                        draw_message(f"Right image: {30 - len(right_coords)} clicks left")

            draw_red_dot(param, (x, y))

    def draw_message(message):
        message_window = np.zeros((100, 800, 3), dtype=np.uint8)
        cv.putText(message_window, message, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        cv.imshow("Message", message_window)

    def draw_red_dot(image_name, point):
        image = left_img if image_name == 'left' else right_img
        cv.circle(image, point, 5, (0, 0, 255), -1)
        cv.imshow("Left Image" if image_name == 'left' else "Right Image", image)

    # Load images
    left_img = cv.imread(left_image_path)
    right_img = cv.imread(right_image_path)

    # Check if images are loaded successfully
    if left_img is None or right_img is None:
        print("Error: Could not read the images.")
        return

    # Display images and collect coordinates
    cv.imshow("Left Image", left_img)
    cv.imshow("Right Image", right_img)

    cv.setMouseCallback("Left Image", on_mouse_click, 'left')
    cv.setMouseCallback("Right Image", on_mouse_click, 'right')

    print("Click 30 times on the left image and 30 times on the right image.")

    while True:
        if len(left_coords) >= 30 and len(right_coords) >= 30:
            break

        key = cv.waitKey(1) & 0xFF
        if key == ord('p'):
            print("Emergency exit triggered.")
            return [], []

    cv.destroyAllWindows()
    return left_coords, right_coords

def compute_xyz(left_coords, right_coords):
    camera_params = {
        "baseline": -94.926,
        "rectified_fx": 648.52,
        "rectified_fy": 648.52,
        "rectified_cx": 635.709,
        "rectified_cy": 370.88,
    }

    baseline = camera_params["baseline"]
    fx = camera_params["rectified_fx"]
    fy = camera_params["rectified_fy"]
    cx = camera_params["rectified_cx"]
    cy = camera_params["rectified_cy"]

    left_xyz = []
    right_xyz = []

    for (ul, vl), (ur, vr) in zip(left_coords, right_coords):
        ucl = ul - cx
        vcl = vl - cy
        ucr = ur - cx
        vcr = vr - cy

        disparity = ucl - ucr

        Z = (fx * baseline) / disparity
        Xl = (ucl * Z) / fx
        Yl = (vcl * Z) / fy
        Xr = (ucr * Z) / fx
        Yr = (vcr * Z) / fy

        left_xyz.append((Xl, Yl, Z))
        right_xyz.append((Xr, Yr, Z))

    print("LEFT IMAGE XYZ || RIGHT IMAGE XYZ")
    for l_xyz, r_xyz in zip(left_xyz, right_xyz):
        print(f"{l_xyz} || {r_xyz}")

if __name__ == "__main__":
    args = read_parse_images()
    left_coords, right_coords = collect_coordinates(args.l_img, args.r_img)
    compute_xyz(left_coords, right_coords)