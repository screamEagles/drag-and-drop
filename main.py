import cv2
from HandTrackingModule import HandDetector
import cvzone  # cvzone version: 1.4.1
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8)
rectangle_colour = (255, 0, 0)
centre_x, centre_y, width, height = 100, 100, 200, 200


class dragRectangle():
    def __init__(self, centre_position, size=[200, 200]):
        self.centre_position = centre_position
        self.size = size

    def update(self, cursor):
        centre_x, centre_y = self.centre_position
        width, height = self.size

        # if the index finger tip is in the rectangle region
        if centre_x - width // 2 < cursor[0] < centre_x + width // 2 and centre_y - height // 2 < cursor[1] < centre_y + height // 2:
            # rectangle_colour = (0, 255, 0)
            self.centre_position = cursor

rectangle_list = []
for x in range(5):
    rectangle_list.append(dragRectangle([x * 250 + 150, 150]))


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    landmark_list, _ = detector.findPosition(img, draw=False)

    if landmark_list:

        length, _, _ = detector.findDistance(8, 12, img, draw=False)
        # print(length)
        if length < 40:
            cursor = landmark_list[8][:2]
            for rectangle in rectangle_list:
                rectangle.update(cursor)

    new_image = np.zeros_like(img, np.uint8)
    for rectangle in rectangle_list:
        centre_x, centre_y = rectangle.centre_position
        width, height = rectangle.size
        cv2.rectangle(new_image, (centre_x - width // 2, centre_y - height // 2), (centre_x + width // 2, centre_y + height // 2), rectangle_colour, cv2.FILLED)

        cvzone.cornerRect(new_image, (centre_x - width // 2, centre_y - height // 2, width, height), 20, rt=0)
    out = img.copy()
    alpha = 0.5
    mask = new_image.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, new_image, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
