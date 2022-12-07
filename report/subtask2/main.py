import numpy as np
import cv2
import argparse
from tqdm import tqdm
from math import sin, cos, atan, floor, pi

def conv2d(img, kernel, fast=True):
    if fast:
        return cv2.filter2D(img, -1, kernel)
    else:
        h = img.shape[0]
        w = img.shape[1]

        by = floor(kernel.shape[0] / 2)
        bx = floor(kernel.shape[1] / 2)

        padded = np.zeros((h + by*2, w + bx*2))
        padded[by:-by, bx:-bx] = img

        result = np.zeros_like(img, dtype=img.dtype)

        for y in range(h):
            for x in range(w):
                result[y, x] = sum(sum(padded[y:y+2*by+1, x:x+2*bx+1] * kernel))

        return result

def houghlines(img, Ts=0.8, Th=210):
    h, w = img.shape

    kernelx = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])

    kernely = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],  
    ])

    # compute partial derivatives in x and y directions
    # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grey = img.astype(np.float32)
    edgex = conv2d(grey, kernelx)
    edgey = conv2d(grey, kernely)

    theta_res = 512
    p_res = 512
    d_theta = 0.005
    p_max = w * cos(atan(-w/h)+pi/2) + h * sin(atan(-w/h)+pi/2)

    H = np.zeros((p_res, theta_res), dtype=np.uint8)
    # for every pixel in the image
    for y in range(h):
        for x in range(w):
            # compute magnitude of gradient
            G = edgex[y, x] * edgex[y, x] + edgey[y, x] * edgey[y, x]
            if G > Ts * Ts:
                G = G ** Ts
                # compute angle of gradient
                phi = atan(edgey[y, x] / edgex[y, x])
                low = int(theta_res * (phi - d_theta) / (2*pi))
                high = int(theta_res * (phi + d_theta) / (2*pi))
                for theta_i in range(low, high):
                    theta = (2*pi * theta_i) / theta_res
                    p = x * cos(theta) + y * sin(theta)
                    p_i = int(p_res * (p+p_max) / (2*p_max))
                    H[p_i, theta_i] = H[p_i, theta_i] + 1

    lines = []

    kw = 5
    kernel = np.ones((kw, kw), dtype=np.uint8)
    kernel[floor(kw/2), floor(kw/2)] = 0

    padded = np.zeros((p_res + kw*2, theta_res + kw*2))
    padded[kw:-kw, kw:-kw] = H

    for theta_i in range(theta_res):
        for p_i in range(p_res):
            if H[p_i, theta_i] > Th:
                if np.max(padded[p_i:p_i+kw, theta_i:theta_i+kw] * kernel) < H[p_i, theta_i]:
                    p = 2*p_max*(p_i / p_res) - p_max
                    theta = 2*pi*(theta_i / theta_res)
                    lines.append((p, theta))
    return (lines, H)

# Compute the Jaccard coeficient between rectangles A and B
# A = (x, y, w, h)
# B = (x, y, w, h)
def J(A, B):
    dx = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0])
    dy = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
    if dx <= 0 or dy <= 0:
        return 0

    intersection = dx * dy

    union = A[2] * A[3] + B[2] * B[3] - intersection

    return intersection / union

# load cascade data from file
cascade_name = "NoEntrycascade/cascade.xml"
model = cv2.CascadeClassifier(cascade_name)

f1s = []
TPRs = []
Recalls = []

for i in tqdm(range(16)):
    # open and preprocess the image
    img_name = "No_entry/NoEntry{}.bmp".format(i)
    frame = cv2.imread(img_name)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # open and parse the ground truth file
    truth_name = "No_entry/NoEntry{}truth.txt".format(i)
    true = []
    with open(truth_name) as f:
        for line in f.readlines():
            l = line.split(" ")
            x = int(l[0])
            y = int(l[1])
            w = int(l[2])
            h = int(l[3])
            true.append((x, y, w, h))

    # number of stop signs actually present in the image
    ground_truth = len(true)

    # detect stop signs
    results = model.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=1, minSize=(8, 8), maxSize=(300, 300))
    filtered = []

    for rect in results:
        region = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()
        if rect[2] != rect[3]:
            d = max(rect[2], rect[3])
            region = cv2.resize(region, (d, d), interpolation=cv2.INTER_LINEAR)
        region = cv2.medianBlur(region, 3)
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(region_hsv, (0, 180, 50), (7, 255, 255))
        mask_red2 = cv2.inRange(region_hsv, (150, 180, 50), (180, 255, 255))

        imask_red1 = mask_red1>0
        imask_red2 = mask_red2>0
        region_red = np.zeros_like(region, np.uint8)
        region_red[imask_red1] = region[imask_red1]
        region_red[imask_red2] = region[imask_red2]

        region_red_gray = cv2.cvtColor(region_red, cv2.COLOR_BGR2GRAY)

        lines = houghlines(region_red_gray, 0.4, 2)[0]
        horizontal_lines = 0
        for p, theta in lines:
            x0 = 0
            y0 = 0
            x1 = 0
            y1 = 0
            if sin(theta) != 0.0:
                x0 = 0
                y0 = p / sin(theta)
                x1 = region.shape[1]
                y1 = p / sin(theta) - (cos(theta)/sin(theta))*x1
            else:
                y0 = 0
                x0 = p / cos(theta)
                y1 = region.shape[1]
                x1 = p / cos(theta) - (sin(theta)/cos(theta))*y1
            if abs(y0 - y1) < region.shape[0] / 2:
                horizontal_lines += 1

        # print(horizontal_lines)
        if horizontal_lines >= 2:
            filtered.append(rect)

    results = filtered
    total_guessed = len(results)

    # check which signs have been found
    found = 0
    for rect in true:
        # draw the true signs
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)
        detected = False
        for guess_rect in results:
            if J(rect, guess_rect) > 0.3:
                detected = True
                break
        if detected:
            found += 1            
        
    precision = 0.0
    if total_guessed != 0:
        precision = found / total_guessed
    recall = found / ground_truth

    f1 = 0.0
    if not (precision == 0.0 and recall == 0.0):
        f1 = 2 * (precision * recall) / (precision + recall)

    f1s.append(f1)
    TPRs.append(precision)
    Recalls.append(recall)
    for rect in results:
        # draw the guesses
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)


    # save hough space image
    if i == 0:
        rect = true[0]
        region = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()
        if rect[2] != rect[3]:
            d = max(rect[2], rect[3])
            region = cv2.resize(region, (d, d), interpolation=cv2.INTER_LINEAR)
        region = cv2.medianBlur(region, 3)
        region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask_red1 = cv2.inRange(region_hsv, (0, 180, 50), (7, 255, 255))
        mask_red2 = cv2.inRange(region_hsv, (150, 180, 50), (180, 255, 255))

        imask_red1 = mask_red1>0
        imask_red2 = mask_red2>0
        region_red = np.zeros_like(region, np.uint8)
        region_red[imask_red1] = region[imask_red1]
        region_red[imask_red2] = region[imask_red2]

        region_red_gray = cv2.cvtColor(region_red, cv2.COLOR_BGR2GRAY)
        H = houghlines(region_red_gray, 0.4, 2)[1]

        cv2.imwrite("houghSpace.png", H)

for f1 in f1s:
    print(f1)

print()

for TPR in TPRs:
    print(TPR)

print()

for recall in Recalls:
    print(recall)