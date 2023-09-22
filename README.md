# CalibrationHub

The CalibrationHub App is a Python application developed using the customtkinter library and OpenCV for calibrating cameras. It allows you to load a set of images of a chessboard pattern, find the chessboard corners, and calibrate the camera. This calibration can be useful for various computer vision tasks like image rectification and 3D reconstruction. Here's how to use the app and what each part of it does:

Features:

#### Image Loading:
Click the "Load Images" button to select a folder containing images of a chessboard pattern.
Thumbnails of the loaded images are displayed in the sidebar.

### Image Selection:

Click on the thumbnails to display the selected image in the main window.
Select the images you want to use for calibration by checking the checkboxes next to the thumbnails.


### Camera Calibration Settings:
Choose the camera model (Normal or Fisheye).
Specify the size of the chessboard squares, width, and height of the chessboard pattern.
Select the unit of measurement for the squares (Millimeter, Centimeter, or Inch).

### Calibration:
After selecting the images and configuring the settings, click the "Start Calibrate" button to begin the calibration process.
The progress of the calibration process is displayed as a progress bar.

### Results:
The calibration results, including camera matrix, distortion coefficients, rotation vectors, and translation vectors, are saved to a YAML file called "camera_calibration.yml."
Reprojection errors for each image are displayed as a bar chart in the "Errors" tab.

### 3D Visualization:
The "Extrinsic Parameters Visualization" tab provides a 3D visualization of the camera extrinsic parameters.
You can view the camera pose in 3D space and visualize how the chessboard pattern and camera models align.

## How to Run

Make sure you have the required Python libraries installed, such as tkinter, numpy, OpenCV, and matplotlib.

Copy and paste the provided Python code into a Python file (e.g., camera_calibration_app.py).

Run the Python file using a Python interpreter (e.g., python camera_calibration_app.py).

Follow the instructions above to load images and perform camera calibration.

1. Click on Load Images button
![fig1](https://github.com/KiaRational/CalibrationHub/blob/main/readme/1.png)
2. Fill the Settings tab based on your situation.
![fig2](https://github.com/KiaRational/CalibrationHub/blob/main/readme/2.png)
3. Select images you want
4. click on Start Calibration button
![fig3](https://github.com/KiaRational/CalibrationHub/blob/main/readme/3.png)
## Dependencies

tkinter: GUI library for creating the graphical user interface.
customtkinter 
numpy: Library for numerical computations.
OpenCV (cv2): Open-source computer vision library for image processing and calibration.
matplotlib: Library for creating data visualizations.

## Author

This Camera Calibration App was created by Kiarational . 
You can contact me at kiarash.ghasemzadeh@gmail.com for any inquiries or contributions.
## License

This app is open-source software distributed under the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT).Feel free to use and modify it for your needs.

