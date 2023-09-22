import numpy as np
import cv2 as cv
import glob
import os
import yaml 
class CameraCalibration:
    def __init__(self, chessboardSize, imageFolderPath, useFisheye=False):
        self.chessboardSize = chessboardSize
        self.imageFolderPath = imageFolderPath
        self.useFisheye = useFisheye
        self.objpoints = []  # 3d point in real-world space
        self.imgpoints = []  # 2d points in image plane.
        self.cameraMatrix = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
    def findChessboardCorners(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        size_of_chessboard_squares_mm = 20
        objp = objp * size_of_chessboard_squares_mm

        images = glob.glob(self.imageFolderPath + '/*.png')

        for image in images:
            if image:
                img = cv.imread(image)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)

                if ret == True:
                    self.objpoints.append(objp)
                    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    self.imgpoints.append(corners)

                    # Draw and save the corners in a results folder
                    cv.drawChessboardCorners(img, self.chessboardSize, corners2, ret)
                    result_folder = os.path.join(os.path.dirname(self.imageFolderPath), "results")
                    os.makedirs(result_folder, exist_ok=True)
                    result_path = os.path.join(result_folder, os.path.basename(image))
                    cv.imwrite(result_path, img)

    def calibrateCamera(self, frameSize):
        if self.useFisheye:
            ret, self.cameraMatrix, self.dist, self.rvecs, self.tvecs = cv.fisheye.calibrate(
                self.objpoints, self.imgpoints, frameSize, None, None)
        else:
            ret, self.cameraMatrix, self.dist, self.rvecs, self.tvecs = cv.calibrateCamera(
                self.objpoints, self.imgpoints, frameSize, None, None)

        # Save camera calibration matrices to YAML files
        calibration_data = {
            'cameraMatrix': self.cameraMatrix,
            'dist': self.dist,
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
        }
        calibration_file = 'camera_calibration.yml'
        with open(calibration_file, 'w') as outfile:
            yaml.dump(calibration_data, outfile)

    def undistortImage(self, inputImagePath, outputImagePath):
        img = cv.imread(inputImagePath)
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.cameraMatrix, self.dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, self.cameraMatrix, self.dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imwrite(outputImagePath, dst)

    def calculateReprojectionError(self):
        mean_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.cameraMatrix, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
            mean_error += error
        return mean_error / len(self.objpoints)
if __name__ == "__main__":
    chessboardSize = (24, 17)
    frameSize = (1440, 1080)
    imageFolderPath = '/home/kia/CalibrationHub/calibration'

    calibration = CameraCalibration(chessboardSize, imageFolderPath, useFisheye=False)
    calibration.findChessboardCorners()
    calibration.calibrateCamera(frameSize)

    reprojection_error = calibration.calculateReprojectionError()
    print("Total reprojection error: {}".format(reprojection_error))

    # Create an "undistorted" folder to save the undistorted images
    undistorted_folder = 'undistorted'
    os.makedirs(undistorted_folder, exist_ok=True)

    # Loop through all images in the folder and undistort them
    images = glob.glob(os.path.join(imageFolderPath, '*.png'))
    for image_path in images:
        filename = os.path.basename(image_path)
        output_image_path = os.path.join(undistorted_folder, filename)
        calibration.undistortImage(image_path, output_image_path)
