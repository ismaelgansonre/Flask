from os import name
import numpy as np
import cv2
import time
import centroidtracker



def read_labels(file_path):
    with open(file_path) as f:
        labels = [line.strip() for line in f]
    return labels


def load_network(config_path, weights_path):
    network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    layers_names_all = network.getLayerNames()
    layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

    return network, layers_names_output


def process_frame(frame, network, layers_names_output, probability_minimum, threshold):
    # Récupérer les dimensions de l'image
    h, w = frame.shape[:2]

    # Obtenir le blob à partir de l'image
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Passer le blob dans le réseau
    network.setInput(blob)
    output_from_network = network.forward(layers_names_output)

    # Obtenir les boîtes englobantes
    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        for detected_objects in result:
            scores = detected_objects[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    # Suppression non maximale
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

    return results, bounding_boxes, confidences, class_numbers


def draw_boxes_and_labels(frame, results, bounding_boxes, confidences, class_numbers, labels, colours):
    rects = []

    if len(results) > 0:
        for i in results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            colour_box_current = colours[class_numbers[i]].tolist()

            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height), colour_box_current, 2)
            rects.append([x_min, y_min, x_min + box_width, y_min + box_height])

            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
            
            cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    return frame, rects


def main():
    camera = cv2.VideoCapture(1)
    ct = centroidtracker.CentroidTracker()
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

     cv2.namedWindow('YOLO v4 Détections en Temps Réel', cv2.WINDOW_NORMAL)
     cv2.imshow('YOLO v4 Détections en Temps Réel', frame)

     if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    camera.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
