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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri



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


def on_mouse_click(event, x, y, flags, param):
    left_img, right_img = param["images"]
    left_coords, right_coords = param["coords"]
    
    if event == cv.EVENT_LBUTTONDOWN:
        if param["side"] == 'left':
            if len(left_coords) >= 30:
                draw_message("You have made your 30 clicks, proceed with the next 30 in the other image")
            else:
                left_coords.append((x, y))
                draw_message(f"Left image: {30 - len(left_coords)} clicks left")
        elif param["side"] == 'right':
            if len(left_coords) < 30:
                draw_message("Please click 30 times on the left image first.")
            else:
                if len(right_coords) >= 30:
                    draw_message("You have made your 30 clicks, proceed with the next steps.")
                else:
                    right_coords.append((x, y))
                    draw_message(f"Right image: {30 - len(right_coords)} clicks left")

        draw_red_dot(param["side"], (x, y), left_img, right_img)


def draw_message(message):
    message_window = np.zeros((100, 800, 3), dtype=np.uint8)
    cv.putText(message_window, message, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("Message", message_window)


def draw_red_dot(image_name, point, left_img, right_img):
    image = left_img if image_name == 'left' else right_img
    cv.circle(image, point, 5, (0, 0, 255), -1)
    cv.imshow("Left Image" if image_name == 'left' else "Right Image", image)


def collect_coordinates(left_image_path: str, right_image_path: str):
    left_coords = []
    right_coords = []

    left_img = cv.imread(left_image_path)
    right_img = cv.imread(right_image_path)

    if left_img is None or right_img is None:
        print("Error: Could not read the images.")
        return [], []

    cv.imshow("Left Image", left_img)
    cv.imshow("Right Image", right_img)

    cv.setMouseCallback("Left Image", on_mouse_click, {"side": 'left', "images": (left_img, right_img), "coords": (left_coords, right_coords)})
    cv.setMouseCallback("Right Image", on_mouse_click, {"side": 'right', "images": (left_img, right_img), "coords": (left_coords, right_coords)})

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
        "baseline": 94.926,
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

    return left_xyz,right_xyz

def print_coordinates(left_coords, right_coords,left_xyz,right_xyz):
    print("LEFT COORDS || RIGHT COORDS")
    for left, right in zip(left_coords, right_coords):
        print(f"{left} || {right}")
    print("LEFT IMAGE XYZ || RIGHT IMAGE XYZ")
    for l_xyz, r_xyz in zip(left_xyz, right_xyz):
        print(f"{l_xyz} || {r_xyz}")
    


def plot_3d_reconstruction(left_xyz, right_xyz):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot left reconstruction
    left_xyz = np.array(left_xyz)
    ax1.scatter(left_xyz[:, 0], left_xyz[:, 1], left_xyz[:, 2], c='r', marker='o')
    ax1.set_title('Left Image 3D Reconstruction')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot right reconstruction
    right_xyz = np.array(right_xyz)
    ax2.scatter(right_xyz[:, 0], right_xyz[:, 1], right_xyz[:, 2], c='b', marker='o')
    ax2.set_title('Right Image 3D Reconstruction')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Create triangulation for left image
    left_tri = mtri.Triangulation(left_xyz[:, 0], left_xyz[:, 1])

    # Plot left surface
    ax1.plot_trisurf(left_tri, left_xyz[:, 2], cmap='viridis', alpha=0.5, edgecolor='black')

    # Create triangulation for right image
    right_tri = mtri.Triangulation(right_xyz[:, 0], right_xyz[:, 1])

    # Plot right surface
    ax2.plot_trisurf(right_tri, right_xyz[:, 2], cmap='viridis', alpha=0.5, edgecolor='black')

    plt.show()

if __name__ == "__main__":
    args = read_parse_images()
    left_coords, right_coords = collect_coordinates(args.l_img, args.r_img)
    left_xyz, right_xyz = compute_xyz(left_coords, right_coords)
    print_coordinates(left_coords, right_coords,left_xyz,right_xyz)
    compute_xyz(left_coords, right_coords)
    plot_3d_reconstruction(left_xyz, right_xyz)