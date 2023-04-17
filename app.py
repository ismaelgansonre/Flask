import uuid
from flask import Flask, Response, render_template, request, redirect, send_file, url_for, send_from_directory
import os
import cv2
import numpy as np
import time

from YoloCam import draw_boxes_and_labels, load_network, process_frame, read_labels
from centroidtracker import CentroidTracker


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
       if 'image' not in request.files:
        return "Aucune image n'a été envoyée.", 400

    image_file = request.files['image']
    image_path = 'uploaded_image.jpg'
    image_file.save(image_path)

    # Charger l'image en tant que BGR
    image_BGR = cv2.imread(image_path)

    # Appeler la fonction de détection d'objets
    result_image_path = detect_objects(image_BGR)

    # Renvoyer l'image résultante
    return send_file(result_image_path, mimetype='image/jpeg')




def detect_objects(image_BGR):
        # Mettez ici votre code pour la détection des objets avec YOLOv4
        # Utilisez self.image_BGR comme image d'entrée
        
        
        h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

        blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                        swapRB=True, crop=False)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'yolo-pou_mou-data', 'obj.names')
        with open(file_path) as f:
            # Getting labels reading every line
            # and putting them into the list
            labels = [line.strip() for line in f]

        config_file_path = os.path.join(base_dir, 'yolo-pou_mou-data', 'new_yolov4-custom.cfg')
        weights_file_path = os.path.join(base_dir, 'yolo-pou_mou-data', 'new_yolov4-custom_best.weights')

        network = cv2.dnn.readNetFromDarknet(config_file_path, weights_file_path)
        #new_yolov4-custom_best
        # Getting list with names of all layers from YOLO v4 network
        layers_names_all = network.getLayerNames()

        layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

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
                cv2.rectangle(image_BGR, (x_min, y_min),
                                        (x_min + box_width, y_min + box_height),
                                        colour_box_current, 2)

                            # Putting text with label and confidence on the original image
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                                confidences[i])
                cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
        print('Total objects been detected:', len(bounding_boxes))
        
        result_image_path = "result_" + str(uuid.uuid4()) + ".jpg"

        cv2.imwrite(result_image_path, image_BGR)
        return result_image_path


def gen_frames():
    camera = cv2.VideoCapture(0)
    ct = CentroidTracker()
    h, w = None, None

    labels_path = 'yolo-pou_mou-data/obj.names'
    config_path = 'yolo-pou_mou-data/new_yolov4-custom.cfg'
    weights_path = 'yolo-pou_mou-data/new_yolov4-custom_best.weights'
    labels = read_labels(labels_path)
    network, layers_names_output = load_network(config_path, weights_path)

    probability_minimum = 0.05
    threshold = 0.5
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    while True:
        _, frame = camera.read()

        if w is None or h is None:
            h, w = frame.shape[:2]

        results, bounding_boxes, confidences, class_numbers = process_frame(
            frame, network, layers_names_output, probability_minimum, threshold)

        frame, rects = draw_boxes_and_labels(
            frame, results, bounding_boxes, confidences, class_numbers, labels, colours)

        objects = ct.update(rects)

        text_number_car_current = 'Nombre de poulets : {}'.format(len(objects.keys()))
        cv2.putText(frame, text_number_car_current, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concatène les informations de l'image

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True, threaded=True,port=8080)
       
    
