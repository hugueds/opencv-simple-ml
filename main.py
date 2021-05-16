import cv2 as cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

events = [i for i in dir(cv) if 'EVENT' in i]

white_frame = np.ones((255, 255, 3), np.uint8) * 255
frame = white_frame.copy()

reference_point = []
cropping = False
circle_color = 0
tree = None
prediction = False
auto_classify = False
frame_counter = 0

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)

colors = [red, green, blue]

color = colors[0]


def draw_circle(event, x, y, flags, params):
    global reference_point, cropping
    if event == cv.EVENT_LBUTTONDOWN:
        if not prediction:
            reference_point.append((x, y, circle_color))
            cv.circle(frame, (x, y), 7, colors[circle_color], -1)
        else:
            p_color = tree.predict([(x, y)])[0]
            cv.circle(frame, (x, y), 7, colors[p_color], -1)


def train():
    global tree
    print('starting training')
    dataset = np.array(reference_point)
    x_train = dataset[:, :2]
    y_train = dataset[:, 2]
    # tree = RandomForestClassifier(n_estimators=10, random_state=0)
    tree = KNeighborsClassifier()
    tree.fit(x_train, y_train)
    print('training finished')


def classify():
    global tree
    x = np.random.randint(255)
    y = np.random.randint(255)
    p_color = tree.predict([(x, y)])[0]
    cv.circle(frame, (x, y), 7, colors[p_color], -1)


cv.namedWindow('main')


while True:

    cv.imshow('main', frame)
    cv.setMouseCallback('main', draw_circle)

    frame_counter += 1
    frame_counter = 0 if frame_counter >= 1000 else frame_counter

    if auto_classify and frame_counter % 25 == 0:
        classify()

    key = cv.waitKey(1)

    if key == ord('1'):
        circle_color = 0

    if key == ord('2'):
        circle_color = 1

    if key == ord('3'):
        circle_color = 2

    if key == ord('t'):
        train()

    if key == ord('c'):
        prediction = False
        if not auto_classify:
            auto_classify = True
        else:
            auto_classify = False

    if key == ord('s'):
        print(reference_point)

    if key == ord('p'):
        if not prediction:
            print('prediction enabled')
        else:
            print('prediction disabled')
        prediction = not prediction

    if key == ord('r'):
        reference_point = []
        frame = white_frame.copy()

    if key & 0xFF == ord('q'):
        break


cv.destroyAllWindows()
