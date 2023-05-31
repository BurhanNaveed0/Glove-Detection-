import math
import cv2
import numpy as np

frame_width = 480
frame_height = 270
frame_x = 0
frame_y = 0
frame_speed = 10

point_x1 = point_y1 = None
point_x2 = point_y2 = None

cam = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorKNN()
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)


def draw_light(contour):
    x, y, w, h = cv2.boundingRect(contour)
    point_x1, point_y1 = (x + int(w / 2), y + int(h / 2))
    cv2.circle(frame, (point_x1, point_y1), 5, (255, 0, 0), -1)
    cv2.putText(frame, ("(" + str(point_x1) + ", " + str(point_y1) + ")"), (point_x1, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    return (point_x1, point_y1)


def draw_distance():
    # Draw line between lights
    cv2.line(frame, (point_x1, point_y1), (point_x2, point_y2), (0, 255, 0), thickness=3)
    # Show midpoint of line
    cv2.circle(frame, (int((point_x1 + point_x2) / 2), int((point_y1 + point_y2) / 2)), 5, (255, 0, 0), -1)
    # Calculator and display distance between lights
    distance = math.sqrt(math.pow((point_x2 - point_x1), 2) + math.pow((point_y2 - point_y1), 2))
    cv2.putText(frame, "Dist: " + str(int(distance)),
                (int((point_x1 + point_x2) / 2), int((point_y1 + point_y2) / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    return distance


while True:
    # Get feed from webcam
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)

    # Remove background
    subtracted = backSub.apply(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prep_frame = cv2.bitwise_and(gray, subtracted, None)
    test = cv2.bitwise_and(gray, subtracted, None)

    # Turn image into gray image
    #prep_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Remove background noise
    prep_frame = cv2.GaussianBlur(src=prep_frame, ksize=(13, 13), sigmaX=0)
    # Create white and black image containing only lights/white objects
    prep_frame = cv2.threshold(prep_frame, 253, 255, cv2.THRESH_BINARY)[1]

    # Shrink detected lights on binary image
    # then smooth out each spot of detected light with a dilation
    prep_frame = cv2.erode(prep_frame, np.ones((3, 3)), 1)
    prep_frame = cv2.dilate(prep_frame, np.ones((9, 9)), 1)

    # Find contours of all white lights/white objects
    unfiltered_contours, _ = cv2.findContours(image=prep_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = []

    # Check shapes of contours and size of contour
    for contour in unfiltered_contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        (x, y, w, h) = cv2.boundingRect(contour)

        # Check if shape is a circle
        if len(approx) > 6 and abs(w - h) < 80 and 100 < cv2.contourArea(contour) < 10000:
            contours.append(contour)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Reset coordinates of lights
    point_x1 = point_y1 = None
    point_x2 = point_y2 = None

    # Draw around the lights
    if len(contours) >= 1 and cv2.contourArea(contours[0]) > 200:
        (point_x1, point_y1) = draw_light(contours[0])
    if len(contours) >= 2 and cv2.contourArea(contours[1]) > 200:
        (point_x2, point_y2) = draw_light(contours[1])

    # Check if both lights have been detected and are of the similar size
    distance = 0
    if (point_x1, point_y1) != (None, None) and (point_x2, point_y2) != (None, None) and abs(
            cv2.contourArea(contours[0]) - cv2.contourArea(contours[1]) < 3000):
        distance = draw_distance()

    # Increase window size if fingers are far apart
    if 200 < distance < 900:
        frame_width *= 1.01
        frame_height *= 1.01

    # Decrease window size if fingers are close
    if 160 > distance > 20:
        frame_width *= 0.99
        frame_height *= 0.99

    # Move window and resize if only one light is present
    print("x", point_x1, point_x2, "y", point_y1, point_y2)
    if point_x1 is not None and point_y1 is not None and point_x2 is None and point_y2 is None:
        if point_x1 > width / 2:
            frame_x += frame_speed
        elif point_x1 < height / 2:
            frame_x -= frame_speed

        if point_y1 > height / 2:
            frame_y += frame_speed
        elif point_y1 < height / 2:
            frame_y -= frame_speed

    # Resize frame to new frame size and move frame
    frame = cv2.resize(frame, (int(frame_width), int(frame_height)))
    cv2.moveWindow("webcam", frame_x, frame_y)

    cv2.imshow('webcam', frame)

    # Exit program when escape key is pressed
    if cv2.waitKey(30) == 27:
        break

cam.release()
cv2.destroyAllWindows()
