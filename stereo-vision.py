"""
José Miguel González Zaragoza
631145 IRSI 6to
UDEM
Dr. Andrés Hernández Gutiérrez

stereo-vision
How to use it:
$ python stereo-vision.py --l_img left_infrared_image.png --r_img right_infrared_image.png
"""

import cv2 as cv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri



def read_parse_images()->argparse.ArgumentParser:
    """
    Parse command-line arguments for left and right image paths.

    Returns:
    argparse.ArgumentParser: Parsed arguments containing paths to left and right images.
    """
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
    """
    Display left and right images.

    Args:
    left_image_path (str): Path to the left image.
    right_image_path (str): Path to the right image.
    """

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
    """
    Handle mouse clicks on the images.

    Args:
    event: The event triggered by the mouse click.
    x (int): The x-coordinate of the mouse click.
    y (int): The y-coordinate of the mouse click.
    flags: Any flags associated with the mouse click event.
    param (dict): Dictionary containing images, coordinates, and side information.
    """

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
    """
    Display a message window.

    Args:
    message (str): The message to be displayed.
    """

    message_window = np.zeros((100, 800, 3), dtype=np.uint8)
    cv.putText(message_window, message, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.imshow("Message", message_window)


def draw_red_dot(image_name, point, left_img, right_img):
    """
    Draw a red dot on the specified image.

    Args:
    image_name (str): Name of the image ('left' or 'right').
    point (tuple): Coordinates of the point where the dot should be drawn.
    left_img: The left image.
    right_img: The right image.
    """

    image = left_img if image_name == 'left' else right_img
    cv.circle(image, point, 5, (0, 0, 255), -1)
    cv.imshow("Left Image" if image_name == 'left' else "Right Image", image)


def collect_coordinates(left_image_path: str, right_image_path: str):
    """
    Collect coordinates from user mouse clicks on left and right images.

    Args:
    left_image_path (str): Path to the left image.
    right_image_path (str): Path to the right image.

    Returns:
    tuple: Lists of left and right coordinates.
    """

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
    """
    Compute 3D coordinates (X, Y, Z) from left and right image coordinates.

    Args:
    left_coords (list): List of left image coordinates.
    right_coords (list): List of right image coordinates.

    Returns:
    tuple: Lists of left and right image 3D coordinates.
    """

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
    """
    Print left and right image coordinates and their corresponding 3D coordinates.

    Args:
    left_coords (list): List of left image coordinates.
    right_coords (list): List of right image coordinates.
    left_xyz (list): List of left image 3D coordinates.
    right_xyz (list): List of right image 3D coordinates.
    """

    print("LEFT COORDS || RIGHT COORDS")
    for left, right in zip(left_coords, right_coords):
        print(f"{left} || {right}")
    print("LEFT IMAGE XYZ || RIGHT IMAGE XYZ")
    for l_xyz, r_xyz in zip(left_xyz, right_xyz):
        print(f"{l_xyz} || {r_xyz}")
    

def plot_3d_reconstruction(left_xyz, right_xyz):
    """
    Plot 3D reconstruction of left and right image points.

    Args:
    left_xyz (list): List of left image 3D coordinates.
    right_xyz (list): List of right image 3D coordinates.
    """

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

    plt.show()


def run_pipeline():
    args = read_parse_images()
    left_coords, right_coords = collect_coordinates(args.l_img, args.r_img)
    left_xyz, right_xyz = compute_xyz(left_coords, right_coords)
    print_coordinates(left_coords, right_coords,left_xyz,right_xyz)
    plot_3d_reconstruction(left_xyz, right_xyz)


if __name__ == "__main__":
    run_pipeline()