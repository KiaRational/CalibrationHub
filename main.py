import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import cv2 as cv
import glob
import os
import yaml 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D toolkit
from matplotlib import cm

class CameraCalibrateApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Viewer App")
        self.geometry(f"{1820}x{780}")
        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure((2, 3 , 4), weight=0)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure((1, 2), weight=1)

        # Create a sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self, width=200)
        # self.sidebar_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        # Create a button to load images
        self.load_button = ctk.CTkButton(self.sidebar_frame, text="Load Images", command=self.load_images)
        self.load_button.grid(row=0, column=0, padx=20, pady=20)
        # Create a frame for image thumbnails
        self.thumbnail_frame = ctk.CTkScrollableFrame(self.sidebar_frame, label_text="Image Thumbnails")
        self.thumbnail_frame.grid(row=2, column=0, rowspan=4, sticky="nsew")

        # Create a frame for displaying the selected image

        self.display_frame = ctk.CTkFrame(self,corner_radius=20,width=700,height=500)
        self.display_frame.grid(row=0, column=1, columnspan=2 , padx=10, pady=20)
        self.selected_image_label = None

        # Initialize image variables
        self.image_paths = []
        self.image_checkboxes = {}  # Dictionary to store image checkboxes

        # create tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=20,width=400)
        self.tabview.grid(row=0, column=3, padx=(10, 10), pady=(20, 20), sticky="nsew")
        self.tabview.add("Errors")
        self.tabview.add("Cam CTE")
        self.tabview.add("Board CTE")

        self.error_canvas = tk.Canvas(self.tabview.tab("Errors"))
        self.error_canvas.pack(fill=tk.BOTH, expand=True)

        self.settings_frame = ctk.CTkTabview(self,corner_radius=20,width= 200,height=500)
        self.settings_frame.add("Settings")
        self.settings_frame.grid(row=0, column=4,padx=10, pady=20)

        self.optionmenu_1 = ctk.CTkOptionMenu(self.settings_frame.tab("Settings"), dynamic_resizing=True,
                                                        values=["Milimeter  .", "Centemeter", "Inch     ."],width=20)
        self.size_checkerboard = ctk.CTkLabel(self.settings_frame.tab("Settings"),text = "Chessboard Squere Size:",width=10)
        self.size_checkerboard_inp = ctk.CTkEntry(self.settings_frame.tab("Settings"),placeholder_text = "number" , width=40)
        self.size_checkerboard_inp.grid(row=0, column=1, padx=0, pady=(10, 10), sticky="ew")
        self.size_checkerboard.grid(row=0, column=0, padx=10, pady=(0, 0), sticky="ew")
        self.optionmenu_1.grid(row=0, column=2, padx=10, pady=(10, 10), sticky="ew")
        self.size_checkerboard_inp.bind("<Return>", self.calibrate)


        self.size_checkerboard2 = ctk.CTkLabel(self.settings_frame.tab("Settings"),text = "Chessboard width Squeres:",width=20)
        self.size_checkerboard_inp2 = ctk.CTkEntry(self.settings_frame.tab("Settings"),placeholder_text = "number" , width=40)
        self.size_checkerboard_inp2.grid(row=1, column=1, padx=10, pady=(10, 10), sticky="ew")
        self.size_checkerboard_inp2.bind("<Return>", self.calibrate)
        self.size_checkerboard2.grid(row=1, column=0, padx=10, pady=(10, 10), sticky="ew")
        self.size_checkerboard3 = ctk.CTkLabel(self.settings_frame.tab("Settings"),text = "Chessboard height Squeres:",width=20)
        self.size_checkerboard_inp3 = ctk.CTkEntry(self.settings_frame.tab("Settings"),placeholder_text = "number" , width=40)
        self.size_checkerboard_inp3.grid(row=2, column=1, padx=10, pady=(0, 0), sticky="ew")
        self.size_checkerboard3.grid(row=2, column=0, padx=10, pady=(10, 10), sticky="ew")
        self.optionmenu_1.bind("<Return>", self.calibrate)

        self.optionmenu_4 = ctk.CTkOptionMenu(self.settings_frame.tab("Settings"), dynamic_resizing=True,
                                                        values=["Normal", "Fisheye"],width=10)
        self.model = ctk.CTkLabel(self.settings_frame.tab("Settings"),text = "Camera Model:",width=40)
        self.optionmenu_4.grid(row=3, column=1, padx=10, pady=(0, 0), sticky="ew")
        self.model.grid(row=3, column=0, padx=10, pady=(10, 10), sticky="ew")
        self.optionmenu_4.bind("<Return>", self.calibrate)

        self.cal_button = ctk.CTkButton(self.settings_frame, text="Start Calibrate", command=self.calibrate)
        self.cal_button.grid(row=3, column=0, padx=20, pady=20)
        self.cal_button.configure(state="disabled")
        self.progress_bar_frame = ctk.CTkFrame(self, width=100)
        self.progress_bar_frame.grid(row=3, column=4, sticky="nsew")
        self.textprogress_bar = ctk.CTkLabel(self.progress_bar_frame , text = "Calibration Progress: ")
        self.progress_bar = ctk.CTkProgressBar(self.progress_bar_frame, orientation="horizontal")
        self.progress_bar.grid(row=0, column=1, columnspan=2,padx=20, pady=20, sticky="ew")
        self.textprogress_bar.grid(row=0,column=0,padx=20)
        self.progress_bar.set(0)  # Set the progress bar to 100% when done

    def load_images(self):
        # Open a file dialog to select a folder containing images
        self.folder_path = filedialog.askdirectory()
        if self.folder_path:
            self.image_paths = [os.path.join(self.folder_path, filename) for filename in os.listdir(self.folder_path) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]


            # Create and display image thumbnails
        for i, image_path in enumerate(self.image_paths):
                img = Image.open(image_path)
                  # Set the thumbnail size

                # Convert the thumbnail to PhotoImage
                thumbnail_img = ctk.CTkImage(img,size=(100,100))

                # Create a label for the thumbnail
                thumbnail_label = ctk.CTkLabel(self.thumbnail_frame,text="", image=thumbnail_img,corner_radius=10)
                thumbnail_label.image = thumbnail_img
                # Create a checkbox for the image
                check_var = ctk.StringVar(value="on")
                checkbox = ctk.CTkCheckBox(self.thumbnail_frame, text=str(i), variable=check_var, onvalue="on", offvalue="off",width=1)
                checkbox.image_path = image_path
                checkbox.grid(row=len(self.image_checkboxes)+1, column=0, padx=10, pady=40, sticky="w")

                self.image_checkboxes[image_path] = check_var  # Store the checkbox state in the dictionary

                # Bind a click event to display the selected image
                thumbnail_label.bind("<Button-1>", lambda event, img_path=image_path: self.display_image(img_path))

                thumbnail_label.grid(row = len(self.image_checkboxes),column=1,padx=0, pady=10, sticky="ew")
        self.cal_button.configure(state="enabled")

    def display_image(self, image_path):
        # Clear the display frame
        for widget in self.display_frame.winfo_children():
            widget.destroy()

        # Load and display the selected image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        img = ctk.CTkImage(img,size=(700,500))
        self.selected_image_label = ctk.CTkLabel(self.display_frame, text="",image=img,corner_radius=10)
        self.selected_image_label.image = img
        self.selected_image_label.pack(fill=ctk.BOTH, expand=True)

    def reload_images(self):
        # Clear the thumbnail frame
        for widget in self.thumbnail_frame.winfo_children():
            widget.destroy()

        # Reload and display image thumbnails for the updated image paths
        for i, image_path in enumerate(self.image_paths):
            img = Image.open(image_path)
            thumbnail_img = ctk.CTkImage(img, size=(100, 100))

            # Create a checkbox for the image with the previous state
            check_var = self.image_checkboxes.get(image_path, ctk.StringVar(value="on"))
            checkbox = ctk.CTkCheckBox(self.thumbnail_frame, text=str(i), variable=check_var, onvalue="on", offvalue="off", width=1)
            checkbox.image_path = image_path
            checkbox.grid(row=i , column=0, padx=10, pady=40, sticky="w")

            self.image_checkboxes[image_path] = check_var  # Store the checkbox state in the dictionary

            # Bind a click event to display the selected image
            thumbnail_label = ctk.CTkLabel(self.thumbnail_frame, text="", image=thumbnail_img, corner_radius=10)
            thumbnail_label.image = thumbnail_img
            thumbnail_label.bind("<Button-1>", lambda event, img_path=image_path: self.display_image(img_path))
            thumbnail_label.grid(row=i, column=1, padx=0, pady=10, sticky="ew")

    def get_selected_images(self):
        self.image_paths = [image_path for image_path, var in self.image_checkboxes.items() if var.get() == "on"]
        # print("Selected Images:", self.selected_images)
        # Add your calibration logic here with the selected images
    def findChessboardCorners(self):
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.chessboardSize[0] * self.chessboardSize[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chessboardSize[0],0:self.chessboardSize[1]].T.reshape(-1,2)
        size_of_chessboard_squares_mm = 20
        objp = objp * size_of_chessboard_squares_mm

        images = glob.glob(self.imageFolderPath + '/*.png')

        total_steps = len(images)
        for i , image in enumerate(self.image_paths):
            if image:
                img = cv.imread(image)
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, self.chessboardSize, None)

                progress_value = (i / total_steps) * 100
                self.progress_bar.set(progress_value)  # Update the progress bar value
                self.update_idletasks()  # Force an update of the GUI to refresh the progress bar
            
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
            print(self.rvecs)

        # Save camera calibration matrices to YAML files
        calibration_data = {
            'cameraMatrix': self.cameraMatrix.tolist(),
            'dist': self.dist.tolist(),
            'rvecs': self.rvecs,
            'tvecs': self.tvecs,
        }
        calibration_file = 'camera_calibration.yml'
        with open(calibration_file, 'w') as outfile:
            yaml.dump(calibration_data, outfile)

    def undistortImage(self, inputImagePath, outputImagePath):
        img = cv.imread(inputImagePath)
        h, w = img.shape[:2]
        self.newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(self.cameraMatrix, self.dist, (w, h), 1, (w, h))
        dst = cv.undistort(img, self.cameraMatrix, self.dist, None, self.newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imwrite(outputImagePath, dst)

    def calculateReprojectionErrors(self):
        errors = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.cameraMatrix, self.dist)
            error = cv.norm(self.imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            errors.append(error)
        return errors

    def calibrate(self, event=None):
        self.useFisheye = (True if str(self.optionmenu_4.get())=="Fisheye" else False)
        self.objpoints = []  # 3d point in real-world space
        self.imgpoints = []  # 2d points in image plane.
        self.cameraMatrix = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.get_selected_images()

        self.chessboardSize = (int(self.size_checkerboard_inp2.get()), int(self.size_checkerboard_inp3.get()))
        frameSize = (1440, 1080)
        self.imageFolderPath = self.folder_path

        self.findChessboardCorners()

        new_folder_name = "results"
        updated_image_paths = []

        for  i , image_path in enumerate(self.image_paths):
            directory, filename = os.path.split(image_path)
            
            # Replace "calibration" with "results" in the directory path
            new_directory = directory.replace("calibration", new_folder_name)
            
            # Create the updated image path
            updated_image_path = os.path.join(new_directory, filename)
            
            # Append the updated path to the list
            updated_image_paths.append(updated_image_path)

            # Append the updated path to the list
            self.image_paths[i] = updated_image_path

        self.reload_images()

        self.progress_bar.set(100)  # Set the progress bar to 100% when done
        self.calibrateCamera(frameSize)

        reprojection_errors = self.calculateReprojectionErrors()

        # Create a bar chart of reprojection errors
        fig_width = 4  # Adjust as needed
        fig_height = 4  # Adjust as needed

        # Create a figure with the specified size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_ylim((0,1))
        ax.bar(range(len(reprojection_errors)), reprojection_errors)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Reprojection Error')
        ax.set_title('Reprojection Errors for Each Image')

        # Create a canvas to embed the matplotlib figure in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.error_canvas)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, padx=20, pady=20)


        # Create an "undistorted" folder to save the undistorted images
        undistorted_folder = 'undistorted'
        os.makedirs(undistorted_folder, exist_ok=True)

        # Loop through all images in the folder and undistort them
        images = glob.glob(os.path.join(self.imageFolderPath, '*.png'))

        for self.image_path in images:
            filename = os.path.basename(self.image_path)
            output_image_path = os.path.join(undistorted_folder, filename)
            self.undistortImage(self.image_path, output_image_path)
        # self.show_extrinsics()

    def show_extrinsics(self):
        # fs = cv2.FileStorage("left_intrinsics.yml", cv2.FILE_STORAGE_READ)
        board_width = int(self.size_checkerboard_inp2.get())
        board_height = int(self.size_checkerboard_inp3.get())
        square_size = int(self.size_checkerboard_inp.get())
        camera_matrix = self.cameraMatrix
        extrinsics = self.rvecs

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_aspect("auto")

        cam_width = 0.064/2  # Set your camera width/2 here
        cam_height = 0.048/2  # Set your camera height/2 here
        scale_focal = 40  # Set your scale_focal value here
        min_values, max_values = self.draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                        scale_focal, extrinsics, board_width,
                                                        board_height, square_size, True)

        X_min = min_values[0]
        X_max = max_values[0]
        Y_min = min_values[1]
        Y_max = max_values[1]
        Z_min = min_values[2]
        Z_max = max_values[2]
        max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0

        mid_x = (X_max+X_min) * 0.5
        mid_y = (Y_max+Y_min) * 0.5
        mid_z = (Z_max+Z_min) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('-y')
        ax.set_title('Extrinsic Parameters Visualization')

        plt.show()
        print('Done')

    def draw_camera_boards(self, ax, camera_matrix, cam_width, cam_height, scale_focal, extrinsics, board_width, board_height, square_size, patternCentric):
            
        min_values = np.zeros((3,1))
        min_values = np.inf
        max_values = np.zeros((3,1))
        max_values = -np.inf

        if patternCentric:
            X_moving = self.create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
            X_static = self.create_board_model(extrinsics, board_width, board_height, square_size)
        else:
            X_static = self.create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
            X_moving = self.create_board_model(extrinsics, board_width, board_height, square_size)
        
        cm_subsection = np.linspace(0.0, 1.0, len(extrinsics))
        colors = [ cm.jet(x) for x in cm_subsection ]

        for i in range(len(X_static)):
            X = np.zeros(X_static[i].shape)
            for j in range(X_static[i].shape[1]):
                X[:,j] = self.transform_to_matplotlib_frame(np.eye(4), X_static[i][:,j])
            ax.plot3D(X[0,:], X[1,:], X[2,:], color='r')
            min_values = np.minimum(min_values, X[0:3,:].min(1))
            max_values = np.maximum(max_values, X[0:3,:].max(1))

        for idx , extrinsic in enumerate(extrinsics):
        
            R, _ = cv.Rodrigues(extrinsic[0:3])
            cMo = np.eye(4,4)
            cMo[0:3,0:3] = R
            # cMo[0:3,3] = extrinsic[0:3]
            for i in range(len(X_moving)):
                X = np.zeros(X_moving[i].shape)
                for j in range(X_moving[i].shape[1]):
                    X[0:4,j] = self.transform_to_matplotlib_frame(cMo, X_moving[i][0:4,j], patternCentric)
                ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[idx])
                min_values = np.minimum(min_values, X[0:3,:].min(1))
                max_values = np.maximum(max_values, X[0:3,:].max(1))

        return min_values, max_values

    def transform_to_matplotlib_frame(self, cMo, X, inverse=False):
        M = np.identity(4)
        M[1,1] = 0
        M[1,2] = 1
        M[2,1] = -1
        M[2,2] = 0

        if inverse:
            return M.dot(self.inverse_homogeneoux_matrix(cMo).dot(X))
        else:
            return M.dot(cMo.dot(X))

    def inverse_homogeneoux_matrix(self, M):
        R = M[0:3, 0:3]
        T = M[0:3, 3]
        M_inv = np.identity(4)
        M_inv[0:3, 0:3] = R.T
        M_inv[0:3, 3] = -(R.T).dot(T)

        return M_inv

    def create_camera_model(self, camera_matrix, width, height, scale_focal, draw_frame_axis=False):
        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1]
        focal = 2 / (fx + fy)
        f_scale = scale_focal * focal

        X_img_plane = np.ones((4,5))
        X_img_plane[0:3,0] = [-width, height, f_scale]
        X_img_plane[0:3,1] = [width, height, f_scale]
        X_img_plane[0:3,2] = [width, -height, f_scale]
        X_img_plane[0:3,3] = [-width, -height, f_scale]
        X_img_plane[0:3,4] = [-width, height, f_scale]

        X_triangle = np.ones((4,3))
        X_triangle[0:3,0] = [-width, -height, f_scale]
        X_triangle[0:3,1] = [0, -2*height, f_scale]
        X_triangle[0:3,2] = [width, -height, f_scale]

        X_center1 = np.ones((4,2))
        X_center1[0:3,0] = [0, 0, 0]
        X_center1[0:3,1] = [-width, height, f_scale]

        X_center2 = np.ones((4,2))
        X_center2[0:3,0] = [0, 0, 0]
        X_center2[0:3,1] = [width, height, f_scale]

        X_center3 = np.ones((4,2))
        X_center3[0:3,0] = [0, 0, 0]
        X_center3[0:3,1] = [width, -height, f_scale]

        X_center4 = np.ones((4,2))
        X_center4[0:3,0] = [0, 0, 0]
        X_center4[0:3,1] = [-width, -height, f_scale]

        X_frame1 = np.ones((4,2))
        X_frame1[0:3,0] = [0, 0, 0]
        X_frame1[0:3,1] = [f_scale/2, 0, 0]

        X_frame2 = np.ones((4,2))
        X_frame2[0:3,0] = [0, 0, 0]
        X_frame2[0:3,1] = [0, f_scale/2, 0]

        X_frame3 = np.ones((4,2))
        X_frame3[0:3,0] = [0, 0, 0]
        X_frame3[0:3,1] = [0, 0, f_scale/2]

        if draw_frame_axis:
            return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
        else:
            return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]

    def create_board_model(self, extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
        width = board_width*square_size
        height = board_height*square_size

        X_board = np.ones((4,5))
        X_board[0:3,0] = [0,0,0]
        X_board[0:3,1] = [width,0,0]
        X_board[0:3,2] = [width,height,0]
        X_board[0:3,3] = [0,height,0]
        X_board[0:3,4] = [0,0,0]

        X_frame1 = np.ones((4,2))
        X_frame1[0:3,0] = [0, 0, 0]
        X_frame1[0:3,1] = [height/2, 0, 0]

        X_frame2 = np.ones((4,2))
        X_frame2[0:3,0] = [0, 0, 0]
        X_frame2[0:3,1] = [0, height/2, 0]

        X_frame3 = np.ones((4,2))
        X_frame3[0:3,0] = [0, 0, 0]
        X_frame3[0:3,1] = [0, 0, height/2]

        if draw_frame_axis:
            return [X_board, X_frame1, X_frame2, X_frame3]
        else:
            return [X_board]

if __name__ == "__main__":
    app = CameraCalibrateApp()
    app.mainloop()
