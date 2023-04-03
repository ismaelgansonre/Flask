

import numpy as np
import cv2
import time


image_BGR = cv2.imread('images\poul_5.jpg')

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
# Pay attention! 'cv2.imshow' takes images in BGR format
cv2.imshow('Original Image', image_BGR)
# Waiting for any key being pressed
cv2.waitKey(0)

h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements




blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)



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


probability_minimum = 0.05
#probability_minimum = 0.00001

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.5
#threshold = 0.0001

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


# Implementing forward pass with our blob and only through output layers
# Calculating at the same time, needed time for forward pass
network.setInput(blob)  # setting blob as input to the network
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

# Showing spent time for forward pass
print('Objects Detection took {:.5f} seconds'.format(end - start))

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

       

        if confidence_current > probability_minimum:
            
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            # Now, from YOLO data format, we can get top left corner coordinates
            # that are x_min and y_min
            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            # Adding results into prepared lists
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)



results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                           probability_minimum, threshold)

# Defining counter for detected objects
counter = 0

# Checking if there is at least one detected object after non-maximum suppression
if len(results) > 0:
    # Going through indexes of results
    for i in results.flatten():
        # Showing labels of the detected objects
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

        # Incrementing counter
        counter += 1

        
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        
        colour_box_current = colours[class_numbers[i]].tolist()

      
        cv2.rectangle(image_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

text_number_car_current = 'Nombre d\'animaux : {}'.format(counter)
cv2.putText(image_BGR, text_number_car_current, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


print()
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)



cv2.imwrite('pred_10.jpg', image_BGR)

cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
cv2.imshow('Detections', image_BGR)
cv2.waitKey(0)
