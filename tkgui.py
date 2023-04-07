import sys
import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from YoloCam import draw_boxes_and_labels, load_network, process_frame, read_labels
from centroidtracker import CentroidTracker

class MainWindow(tk.Tk):
    
    def __init__(self):
        super().__init__()

        self.title("Détection de poulets")
        self.geometry("700x800")

        self.image_frame = tk.Frame(self)
        self.image_frame.pack(side=tk.TOP, pady=(50, 20), expand=True)

        self.label_image_original = tk.Label(self.image_frame)
        self.label_image_original.pack(side=tk.LEFT)

        self.label_image_result = tk.Label(self.image_frame)
        self.label_image_result.pack(side=tk.LEFT)

        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.TOP, anchor=tk.CENTER)

        self.button_load_image = tk.Button(self.button_frame, text="Charger une image", command=self.load_image)
        self.button_load_image.pack(side=tk.LEFT)

        self.button_detect_objects = tk.Button(self.button_frame, text="Détecter les objets", command=self.detect_objects)
        self.button_detect_objects.pack(side=tk.LEFT)

        self.button_open_new_window = tk.Button(self.button_frame, text="Ouvrir la nouvelle fenêtre", command=self.open_new_window)
        self.button_open_new_window.pack(side=tk.LEFT)

    def open_new_window(self):
      self.new_window = NewWindow(self)
      self.new_window.protocol("WM_DELETE_WINDOW", self.new_window.onClose)
      self.new_window.mainloop()


    def display_image_original(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (400, 400))
        image_pil = Image.fromarray(image_resized)
        image_tk = ImageTk.PhotoImage(image_pil, master=self.label_image_original)
        self.label_image_original.config(image=image_tk)
        self.label_image_original.image = image_tk

    def display_image_result(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (400, 400))
        image_pil = Image.fromarray(image_resized)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.label_image_result.config(image=image_tk)
        self.label_image_result.image = image_tk

    def load_image(self):
        file_name = filedialog.askopenfilename(title="Ouvrir une image", filetypes=[("Images", "*.png *.xpm *.jpg *.bmp"), ("Tous les fichiers", "*.*")])
        if file_name:
            self.image_BGR = cv2.imread(file_name)
            self.display_image_original(self.image_BGR)

    def detect_objects(self):
        # Mettez ici votre code pour la détection des objets avec YOLOv4
        # Utilisez self.image_BGR comme image d'entrée pass
         
        h, w = self.image_BGR.shape[:2]  # Slicing from tuple only first two elements

        blob = cv2.dnn.blobFromImage(self.image_BGR, 1 / 255.0, (416, 416)
        ,swapRB=True, crop=False)
        
        with open('yolo-pou_mou-data\obj.names') as f:
            # Getting labels reading every line
            # and putting them into the list
            labels = [line.strip() for line in f]

        network = cv2.dnn.readNetFromDarknet('yolo-pou_mou-data/new_yolov4-custom.cfg',
                                            'yolo-pou_mou-data/new_yolov4-custom_best.weights')
        #new_yolov4-custom_best
        # Getting list with names of all layers from YOLO v4 network
        layers_names_all = network.getLayerNames()

        layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

        probability_minimum = 0.07
        #probability_minimum = 0.00001

        # Setting threshold for filtering weak bounding boxes
        # with non-maximum suppression
        threshold = 0.5


        # Generating colours for representing every detected object
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

        # Implementing Forward pass

        network.setInput(blob)  # setting blob as input to the network
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

        # Showing spent time for single current frame
        print('Current frame took {:.5f} seconds'.format(end - start))

        # Preparing lists for detected bounding boxes,
        # obtained confidences and class's number
        bounding_boxes = []
        confidences = []
        class_numbers = []

        # Going through all output layers after feed forward pass
        for result in output_from_network:



            # Going through all detections from current output layer
            for detected_objects in result:

                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]

                # Eliminating weak predictions with minimum probability
                if confidence_current > probability_minimum:

                    # Scaling bounding box coordinates to the initial image size
                    # YOLO data format keeps coordinates for center of bounding box
                    # and its current width and height
                    # That is why we can just elementwise multiply them elementwise
                    # with height and width of the original image and in this way get
                    # coordinates for center of bounding box, its width and height for original image
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min,
                                        int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
        probability_minimum, threshold)
        counter = 0
                    
                    # Checking if there is at least one detected object after non-maximum suppression
        if len(results) > 0:

                        # Going through indexes of results
            for i in results.flatten():

                            # Showing labels of the detected objects
                print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
                            # Incrementing counter
                counter += 1

                            # Getting current bounding box coordinates,
                            # its width and height
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                            # Preparing colour for current bounding box
                            # and converting from numpy array to list
                colour_box_current = colours[class_numbers[i]].tolist()

                            # Drawing bounding box on the original image
                cv2.rectangle(self.image_BGR, (x_min, y_min),
                                        (x_min + box_width, y_min + box_height),
                                        colour_box_current, 2)

                            # Putting text with label and confidence on the original image
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                                confidences[i])
                cv2.putText(self.image_BGR, text_box_current, (x_min, y_min - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
        print('Total objects been detected:', len(bounding_boxes))
        self.display_image_result(self.image_BGR)
class NewWindow(tk.Toplevel):
    def __init__(self, parent=None):
        super(NewWindow, self).__init__(parent)
        
        self.title("Nouvelle Fenêtre")
        self.geometry("800x600")

        self.video_frame = tk.Frame(self)
        self.video_frame.pack()

        self.label_video = tk.Label(self.video_frame)
        self.label_video.pack()
        
        self.label_nombre_poulets = tk.Label(self.video_frame, text="Nombre de poulets : 0")
        self.label_nombre_poulets.pack(side=tk.TOP, anchor=tk.CENTER)
        # Charger le réseau YOLO et ses paramètres
        labels_path = 'yolo-pou_mou-data/obj.names'
        config_path = 'yolo-pou_mou-data/new_yolov4-custom.cfg'
        weights_path = 'yolo-pou_mou-data/new_yolov4-custom_best.weights'
        self.labels = read_labels(labels_path)
        self.network, self.layers_names_output = load_network(config_path, weights_path)

        self.probability_minimum = 0.05
        self.threshold = 0.5
        self.colours = np.random.randint(0, 255, size=(len(self.labels), 3), dtype='uint8')

        # Ouvrir la caméra
        self.camera = cv2.VideoCapture(1)
        
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        self.update_video_frame()
        
    def update_video_frame(self):
        ct = CentroidTracker()
        ret, frame = self.camera.read()
        
        if ret:
          results, bounding_boxes, confidences, class_numbers = process_frame(
            frame, self.network, self.layers_names_output, self.probability_minimum, self.threshold)

        frame, _ = draw_boxes_and_labels(
            frame, results, bounding_boxes, confidences, class_numbers, self.labels, self.colours)
        objects = ct.update(_)
        text_number_car_current = 'Nombre de poulets : {}'.format(len(objects.keys()))
        cv2.putText(frame, text_number_car_current, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        cv2.namedWindow('YOLO v4 Détections en Temps Réel', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO v4 Détections en Temps Réel', frame)


        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame)
        frame_tk = ImageTk.PhotoImage(frame_pil)

        self.label_video.config(image=frame_tk)
        self.label_video.image = frame_tk
        self.label_nombre_poulets.config(text="Nombre de poulets : {}".format(text_number_car_current))

        self.after(30, self.update_video_frame)


    def onClose(self):
        self.cap.release()
        self.destroy()
if __name__ == '__main__':
    window = MainWindow()
    window.mainloop()
