import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2

class ImageViewerApp(ctk.CTk):
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
        self.display_frame.grid(row=0, column=1, columnspan=2 , padx=20, pady=20)
        self.selected_image_label = None

        # Initialize image variables
        self.image_paths = []

        # create tabview
        self.tabview = ctk.CTkTabview(self, corner_radius=20,width=400)
        self.tabview.grid(row=0, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.tabview.add("CTkTabview")
        self.tabview.add("Tab 2")
        self.tabview.add("Tab 3")
        self.tabview.tab("CTkTabview").grid_columnconfigure(0, weight=1)  # configure grid of individual tabs
        self.tabview.tab("Tab 2").grid_columnconfigure(0, weight=1)
        self.settings_frame = ctk.CTkTabview(self,corner_radius=20,width= 100,height=500)
        self.settings_frame.add("Settings")
        self.settings_frame.grid(row=0, column=4,padx=20, pady=20)

        self.optionmenu_1 = ctk.CTkOptionMenu(self.settings_frame.tab("Settings"), dynamic_resizing=True,
                                                        values=["Milimeter", "Centemeter", "Inch"],width=10)
        self.size_checkerboard = ctk.CTkLabel(self.settings_frame.tab("Settings"),text = "Checkerboard Size:",width=20)
        self.size_checkerboard_inp = ctk.CTkEntry(self.settings_frame.tab("Settings"),placeholder_text = "number" , width=40)
        self.size_checkerboard_inp.grid(row=0, column=1, padx=20, pady=(10, 10))
        self.size_checkerboard.grid(row=0, column=0, padx=20, pady=(10, 10))
        self.optionmenu_1.grid(row=0, column=2, padx=20, pady=(10, 10))

        self.optionmenu_2 = ctk.CTkOptionMenu(self.settings_frame.tab("Settings"), dynamic_resizing=True,
                                                        values=["Milimeter", "Centemeter", "Inch"],width=10)
        self.size_checkerboard2 = ctk.CTkLabel(self.settings_frame.tab("Settings"),text = "Checkerboard Size:",width=20)
        self.size_checkerboard_inp2 = ctk.CTkEntry(self.settings_frame.tab("Settings"),placeholder_text = "number" , width=40)
        self.size_checkerboard_inp2.grid(row=1, column=1, padx=20, pady=(10, 10))
        self.size_checkerboard2.grid(row=1, column=0, padx=20, pady=(10, 10))
        self.optionmenu_2.grid(row=1, column=2, padx=20, pady=(10, 10))
    def load_images(self):
        # Open a file dialog to select a folder containing images
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.image_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]


            # Create and display image thumbnails
            for image_path in self.image_paths:
                img = Image.open(image_path)
                  # Set the thumbnail size

                # Convert the thumbnail to PhotoImage
                thumbnail_img = ctk.CTkImage(img,size=(100,100))

                # Create a label for the thumbnail
                thumbnail_label = ctk.CTkLabel(self.thumbnail_frame,text="", image=thumbnail_img,corner_radius=10)
                thumbnail_label.image = thumbnail_img

                # Bind a click event to display the selected image
                thumbnail_label.bind("<Button-1>", lambda event, img_path=image_path: self.display_image(img_path))

                thumbnail_label.pack(padx=10, pady=10, side=ctk.TOP, anchor=ctk.N)

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

if __name__ == "__main__":
    app = ImageViewerApp()
    app.mainloop()
