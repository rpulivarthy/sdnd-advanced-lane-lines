import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import pickle

def calibrate(path, out_dir):
    images = glob.glob(path)
    obj_points=[]
    img_points=[]
    for filename in images:
        for nx in [6, 7, 8, 9]:
            for ny in [5, 6]:
                img = cv2.imread(filename)
                # Convert to gray
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Find the chessboard corners
                ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
                obj_point = np.zeros((nx*ny, 3), np.float32)
                # Transform this to 2x2 array
                obj_point[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
                # If successful in finding cornere
                if ret:
                    obj_points.append(obj_point)
                    img_points.append(corners)
                    chess_board_corners= cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    image_name = os.path.split(filename)[1]
                    out_image_name = out_dir + 'nx_' +str(nx) +'ny_' + str(ny) + '_' + image_name
                    cv2.imwrite(out_image_name, img)
                    break
        cv2.destroyAllWindows()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return mtx, dist

out_dir = 'output_images/corners_found/'
path = os.path.join("camera_cal", "*.jpg")
mtx, dist = calibrate(path, out_dir)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_dist_pickle.p", "wb" ) )
print(mtx, dist)