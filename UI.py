import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from YoloImg import YoloImg

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Détection d'objets avec YOLO")
        self.geometry("800x600")

        self.yolo = YoloImg()

        self.create_widgets()

    def create_widgets(self):
        self.btn_choisir_image = tk.Button(self, text="Choisir une image", command=self.load_image)
        self.btn_choisir_image.pack(pady=10)

        self.canvas = tk.Canvas(self, width=600, height=400)
        self.canvas.pack(pady=20)

        self.btn_detecter = tk.Button(self, text="Détecter les objets", command=self.detect_objects)
        self.btn_detecter.pack(pady=10)

        self.btn_afficher= tk.Button(self, text="Afficher les résultats", command=self.show_results)
        self.btn_afficher.pack(pady=10)
        
    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image.thumbnail((600, 400))
            self.image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)

    def detect_objects(self):
        if hasattr(self, 'image_path'):
            result_image_path = self.yolo.detect_objects(self.image_path)
            image = Image.open(result_image_path)
            image.thumbnail((600, 400))
            self.image = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
