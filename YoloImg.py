import sys
import numpy as np
import cv2
import time
from PyQt5.QtWidgets import QHBoxLayout, QSlider, QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QGraphicsPixmapItem, QGraphicsScene
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QGraphicsPixmapItem, QGraphicsScene,QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QHBoxLayout
import tkinter as tk
from tkinter import filedialog
import cv2

import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms'



# Modifiez la classe MainWindow comme suit
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Détection de poulets")
        self.setGeometry(100, 100, 1200, 600)

        layout = QVBoxLayout()

        self.image_layout = QHBoxLayout()

        self.label_image_original = QLabel(self)
        self.image_layout.addWidget(self.label_image_original)

        self.label_image_result = QLabel(self)
        self.image_layout.addWidget(self.label_image_result)

        layout.addLayout(self.image_layout)
#Section Bouton
        self.button_load_image = QPushButton("Charger une image", self)
        self.button_load_image.clicked.connect(self.load_image)
        layout.addWidget(self.button_load_image)

        self.button_detect_objects = QPushButton("Détecter les objets", self)
        self.button_detect_objects.clicked.connect(self.detect_objects)
        layout.addWidget(self.button_detect_objects)

        self.button_open_new_window = QPushButton("Ouvrir la nouvelle fenêtre", self)
        self.button_open_new_window.clicked.connect(self.open_new_window)
        layout.addWidget(self.button_open_new_window)

        # Ajoutez le slider ici
        # Afficher le titre du slider
        """ self.label_slider = QLabel("Probabilité minimale :", self, )
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_probability_minimum)


        layout.addWidget(self.label_slider, 0, Qt.AlignLeft)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.update_threshold) """

        """ layout.addWidget(self.label_slider, 0, Qt.AlignLeft)
        layout.addWidget(self.slider) """

        #self.slider.valueChanged.connect(self.detect_objects_with_updated_probability)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    # Autres méthodes ...
    def open_new_window(self):
        self.new_window = NewWindow(self)
        self.new_window.show()
    def display_image_original(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        image_qt = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image_qt)
        self.label_image_original.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    def display_image_result(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        image_qt = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image_qt)
        self.label_image_result.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

    # Modifiez la méthode load_image comme suit
    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Ouvrir une image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_BGR = cv2.imread(file_name)
            self.display_image_original(self.image_BGR)

    def update_probability_minimum(self):
        self.probability_minimum = self.slider.value() / 100
    def update_threshold(self):
        self.threshold = self.slider.value() / 100
    

    def detect_objects_with_updated_probability(self):
        self.update_probability_minimum()
        self.detect_objects()
    def detect_objects_with_updated_threshold(self):
        self.update_threshold()
        self.detect_objects()
      #Fonction de detection des objets

    def detect_objects(self):
        # Mettez ici votre code pour la détection des objets avec YOLOv4
        # Utilisez self.image_BGR comme image d'entrée
        
        
        h, w = self.image_BGR.shape[:2]  # Slicing from tuple only first two elements

        blob = cv2.dnn.blobFromImage(self.image_BGR, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)
        
        with open('yolo-pou_mou-data/obj.names') as f:
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

class NewWindow(QDialog):
    def __init__(self, parent=None):
        super(NewWindow, self).__init__(parent)

        self.setWindowTitle("Nouvelle Fenêtre")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout()

        # Ajoutez ici les widgets et les fonctionnalités pour l'extraction et la caractérisation des objets
        label = QLabel("Ici, vous pouvez extraire et caractériser les objets.")
        layout.addWidget(label)

        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



