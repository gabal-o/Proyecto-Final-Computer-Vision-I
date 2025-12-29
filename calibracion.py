from typing import List
import numpy as np
import cv2
import copy  
import glob
import os

def show_image(name,img):
    cv2.imshow("drawchessboard"+name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

def get_chessboard_points(chessboard_shape, dx, dy):
    cols, rows = chessboard_shape
    vector = []
    for row in range(rows):
        for col in range(cols):
            x = col*dx
            y = row*dy
            z = 0
            vector.append([float(x),float(y),float(z)])
    objp = np.asarray(vector, dtype=np.float32)
    return objp.reshape(-1, 1, 3)

def write_image(name, img):
    cv2.imwrite(f"drawchessboard_{name}.jpg", img)

def calibracion():
    imgs_path = [item for item in glob.glob("ImagenesCalibracion/*.jpg")]
    imgs = load_images(imgs_path)
    corners = [cv2.findChessboardCorners(img, (7,9)) for img in imgs]

    corners_copy = copy.deepcopy(corners)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)

    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]

    corners_refined = [cv2.cornerSubPix(i, cor[1], (7, 9), (-1, -1), criteria) if cor[0] else [] for i, cor in zip(imgs_gray, corners_copy)]

    imgs_copy = copy.deepcopy(imgs)

    imgs_draw = []
    for i in range(len(imgs)):
        img_draw = cv2.drawChessboardCorners(imgs[i], (7,9),  corners[i][1], corners[i][0])
        imgs_draw.append(img_draw)



    chessboard_points = get_chessboard_points((7, 9), 20, 20)
    objpoints = []
    for _ in range(len(imgs)):
        objpoints.append(chessboard_points)
    np.asarray(objpoints) 

    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)
    # TODO
    cameraMat = np.zeros((3,3))
    distcoef = np.zeros((1,4))
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(np.asarray(objpoints),valid_corners,imgs[0].shape[:2],cameraMat,distcoef,criteria=criteria)
    # Obtain extrinsics
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))
    print(extrinsics)
    # Print outputs
    print("Intrinsics:\n", intrinsics)
    print("Distortion coefficients:\n", dist_coeffs)
    print("Root mean squared reprojection error:\n", rms)


calibracion()