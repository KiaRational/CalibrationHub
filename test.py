import os
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

class ImageViewerApp(ctk.CTk):
    def __init__(self, folder_path):
        super().__init__()

        self.title("Image Viewer")
        self.geometry("1000x600")

        self.load_and_display_images(folder_path)

    def load_images_from_folder(self, folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        return image_files

    def display_large_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((int(img.size[0] // 2.5), int(img.size[1] // 2.5)))  # Resize to the desired size
        img = ImageTk.PhotoImage(img)
        self.large_img_label.configure(image=img)
        self.large_img_label.image = img

    def load_and_display_images(self, folder_path):
        image_files = self.load_images_from_folder(folder_path)

        frame = ctk.CTkFrame(self)
        frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        small_images_frame = ctk.CTkScrollableFrame(frame, label_text="Images")
        small_images_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True)

        self.large_image_frame = ctk.CTkFrame(frame)
        self.large_image_frame.pack(side=ctk.RIGHT, fill=ctk.BOTH, expand=True)

        self.large_img_label = ctk.CTkLabel(self.large_image_frame)
        self.large_img_label.pack(fill=ctk.BOTH, expand=True)

        for file_name in image_files:
            image_path = os.path.join(folder_path, file_name)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img.thumbnail((100, 100))
            img = ImageTk.PhotoImage(img)

            label = ctk.CTkLabel(small_images_frame, image=img, text=file_name, compound=ctk.TOP)
            label.image = img

            label.bind("<Button-1>", lambda event, img_path=image_path: self.display_large_image(img_path))

            label.pack(padx=10, pady=10, side=ctk.TOP, anchor=ctk.N)

if __name__ == "__main__":
    folder_path = "/home/kia/CalibrationHub/calibration"
    app = ImageViewerApp(folder_path)
    app.mainloop()


