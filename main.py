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

# perform a hough transform to find circles in img
# min_r is the small radius of circle that will be searched for
# max_r is the largest radius of circle that will be searched for
# Ts is the threshold applied to the magnitude of the gradient, should be in the range 0-1
# Th is the threshold used to determine if a circle is found
def houghcircles(img, min_r, max_r, Ts=0.5, Th=20, fast=True):
    if fast:
        return cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1, param1=int(Ts*255), param2=Th, minRadius=min_r, maxRadius=max_r)[0]
    else:
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

        H = np.zeros((h, w, max_r - min_r), dtype=np.uint8)
        # for every pixel in the image
        for y in tqdm(range(h)):
            for x in range(w):
                # compute magnitude of gradient
                G = edgex[y, x] * edgex[y, x] + edgey[y, x] * edgey[y, x]
                if G > Ts * Ts:
                    G = G ** Ts
                    # compute angle of gradient
                    p = atan(edgey[y, x] / edgex[y, x])
                    # for all possible radii increment count in H
                    for r in range(min_r, max_r):
                        x0 = int(x + r * cos(p))
                        y0 = int(y + r * sin(p))

                        if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
                            continue

                        H[y0, x0, r - min_r] = H[y0, x0, r - min_r] + 1

                        x0 = int(x - r * cos(p))
                        y0 = int(y - r * sin(p))

                        if x0 < 0 or x0 >= w or y0 < 0 or y0 >= h:
                            continue

                        H[y0, x0, r - min_r] = H[y0, x0, r - min_r] + 1

        circles = []

        kw = 5
        kernel = np.ones((kw, kw, kw), dtype=np.uint8)
        kernel[floor(kw/2), floor(kw/2), floor(kw/2)] = 0
        d = max_r - min_r

        padded = np.zeros((h + kw*2, w + kw*2, d + kw*2))
        padded[kw:-kw, kw:-kw, kw:-kw] = H

        # for all possible circles
        for y0 in tqdm(range(h)):
            for x0 in range(w):
                for r in range(min_r, max_r):
                    z = r - min_r
                    # if greater than threshold
                    if H[y0, x0, z] > Th:
                        # and also local maxima
                        if np.max(padded[y0:y0+kw, x0:x0+kw, z:z+kw] * kernel) < H[y0, x0, z]:
                            # probably a circle here
                            circles.append((x0, y0, r))
        return circles

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

parser = argparse.ArgumentParser()
parser.add_argument('-num', '-n', type=int, default=0)
args = parser.parse_args()
img_num = args.num

# load cascade data from file
cascade_name = "NoEntrycascade/cascade.xml"
model = cv2.CascadeClassifier(cascade_name)


# circle detection
# img = cv2.imread("no_entry.jpg")
# grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# grey = cv2.medianBlur(grey, 5)

# circles = houghcircles(grey, 20, 500, Ts=0.5, Th=150, fast=True)

# print(len(circles))

# for circle in circles:
#     cv2.circle(img, (int(circle[0]), int(circle[1])), int(circle[2]), (0, 255, 0), 2)

# cv2.imwrite("detected.jpg", img)

# best_f1 = 0
# bestf1Recall = 0
# bestf1Precision = 0
# bestRecall = 0
# bestf1ScaleFactor = 0
# bestf1MinNeighbors = 0

# best_recall = 0
# bestRecalF1 = 0
# bestRecalPrecision = 0
# bestRecallScaleFactor = 0
# bestRecallMinNeighbors = 0

# best_precision = 0
# bestPrecisionF1 = 0
# bestPrecisionRecall = 0
# bestPrecisionScaleFactor = 0
# bestPrecisionMinNeighbors = 0

# scaleFactor = 1.01
# for _ in tqdm(range(50)):
#     for minNeighbors in range(5):
#         ground_truth = 0
#         found = 0
#         total_guessed = 0

#         for i in range(16):
#             img_name = "No_entry/NoEntry{}.bmp".format(i)
#             frame = cv2.imread(img_name)
#             frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame_gray = cv2.equalizeHist(frame_gray)

#             truth_name = "No_entry/NoEntry{}truth.txt".format(i)
#             true = []
#             with open(truth_name) as f:
#                 for line in f.readlines():
#                     l = line.split(" ")
#                     x = int(l[0])
#                     y = int(l[1])
#                     w = int(l[2])
#                     h = int(l[3])
#                     true.append((x, y, w, h))

#             ground_truth += len(true)

#             results = model.detectMultiScale(frame_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(8, 8), maxSize=(300, 300))
#             total_guessed += len(results)
            
#             for true_rect in true:
#                 detected = False
#                 for guess_rect in results:
#                     if J(true_rect, guess_rect) > 0.3:
#                         detected = True
#                         break
#                 if detected:
#                     found += 1            
                
#         if total_guessed == 0:
#             continue
        
#         if found == 0:
#             continue

#         precision = found / total_guessed
#         recall = found / ground_truth

#         f1 = 2 * (precision * recall) / (precision + recall)
#         if f1 > best_f1:
#             best_f1 = f1
#             bestf1Recall = recall
#             bestf1Precision = precision
#             bestf1ScaleFactor = scaleFactor
#             bestf1MinNeighbors = minNeighbors
        
#         if recall > best_recall:
#             best_recall = recall
#             bestRecalF1 = f1
#             bestRecalPrecision = precision
#             bestRecallScaleFactor = scaleFactor
#             bestRecallMinNeighbors = minNeighbors

#         if precision > best_precision:
#             best_precision = precision
#             bestPrecisionF1 = f1
#             bestPrecisionRecall = recall
#             bestPrecisionScaleFactor = scaleFactor
#             bestPrecisionMinNeighbors = minNeighbors
#     scaleFactor += 0.02
    
# print(best_f1)
# print(bestf1Recall)
# print(bestf1Precision)
# print(bestf1ScaleFactor)
# print(bestf1MinNeighbors)
# print()
# print(bestRecalF1)
# print(best_recall)
# print(bestRecalPrecision)
# print(bestRecallScaleFactor)
# print(bestRecallMinNeighbors)
# print()
# print(bestPrecisionF1)
# print(bestPrecisionRecall)
# print(best_precision)
# print(bestPrecisionScaleFactor)
# print(bestPrecisionMinNeighbors)


# img_name = "No_entry/NoEntry{}.bmp".format(img_num)
# frame = cv2.imread(img_name)
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# frame_gray = cv2.equalizeHist(frame_gray)
# results = model.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=0, minSize=(8, 8), maxSize=(300, 300))

# print(len(results))

# for rect in results:
#     s = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
#     if rect[2] != rect[3]:
#         d = max(rect[2], rect[3])
#         s = cv2.resize(s, (d, d), interpolation=cv2.INTER_LINEAR)
#     s_hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
#     mask_red1 = cv2.inRange(s_hsv, (0, 180, 50), (12, 255, 255))
#     mask_red2 = cv2.inRange(s_hsv, (150, 180, 50), (180, 255, 255))

#     imask_red1 = mask_red1>0
#     imask_red2 = mask_red2>0
#     s_red = np.zeros_like(s, np.uint8)
#     s_red[imask_red1] = s[imask_red1]
#     s_red[imask_red2] = s[imask_red2]

#     cv2.imshow("t", s_red)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     # cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 0, 255), 2)

# cv2.imwrite("detected.jpg", frame)

# circles = houghcircles(frame_gray, 10, 250)
# for x, y, r in circles:
#     cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

# lines = houghlines(frame_gray)
# for p, theta in lines:
#     if sin(theta) != 0.0:
#         x0 = 0
#         y0 = p / sin(theta)
#         x1 = frame.shape[1]
#         y1 = p / sin(theta) - (cos(theta)/sin(theta))*x1
#         print((int(x0), int(y0)), (int(x1), int(y1)))
#         cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
#     else:
#         y0 = 0
#         x0 = p / cos(theta)
#         y1 = frame.shape[1]
#         x1 = p / cos(theta) - (sin(theta)/cos(theta))*y1
#         cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

# cv2.imwrite("detected.jpg", frame)

# def save_grad(i):
#     img_name = "No_entry/NoEntry{}.bmp".format(i)
#     frame = cv2.imread(img_name)
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frame_gray = cv2.equalizeHist(frame_gray)

#     truth_name = "No_entry/NoEntry{}truth.txt".format(i)
#     true = []
#     with open(truth_name) as f:
#         for line in f.readlines():
#             l = line.split(" ")
#             x = int(l[0])
#             y = int(l[1])
#             w = int(l[2])
#             h = int(l[3])
#             true.append((x, y, w, h))

#     rect = true[0]

#     region = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()
#     if rect[2] != rect[3]:
#         d = max(rect[2], rect[3])
#     region = cv2.resize(region, (d, d), interpolation=cv2.INTER_LINEAR)
#     region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
#     mask_red1 = cv2.inRange(region_hsv, (0, 180, 50), (7, 255, 255))
#     mask_red2 = cv2.inRange(region_hsv, (150, 180, 50), (180, 255, 255))

#     imask_red1 = mask_red1>0
#     imask_red2 = mask_red2>0
#     region_red = np.zeros_like(region, np.uint8)
#     region_red[imask_red1] = region[imask_red1]
#     region_red[imask_red2] = region[imask_red2]

#     region_red_gray = cv2.cvtColor(region_red, cv2.COLOR_BGR2GRAY)
#     region_red_gray = cv2.medianBlur(region_red_gray, 5)

#     h, w = region_red_gray.shape

#     kernelx = np.array([
#         [-1, 0, 1],
#         [-2, 0, 2],
#         [-1, 0, 1],
#     ])

#     kernely = np.array([
#         [-1, -2, -1],
#         [0, 0, 0],
#         [1, 2, 1],  
#     ])

#     # compute partial derivatives in x and y directions
#     # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
#     grey = region_red_gray.astype(np.float32)
#     edgex = conv2d(grey, kernelx)
#     edgey = conv2d(grey, kernely)

#     G = np.sqrt(edgex * edgex + edgey * edgey)

#     cv2.imwrite("grad{}.png".format(i), G)

#     lines, H = houghlines(region_red_gray, 0.4, 2)

#     cv2.imwrite("hough{}.png".format(i), H)

        
# save_grad(2)
# save_grad(3)
    

cascade_name = "NoEntrycascade/cascade.xml"
model = cv2.CascadeClassifier(cascade_name)

f1s = []
TPRs = []

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

    # for rect in results:
    #     region = frame[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()
    #     if rect[2] != rect[3]:
    #         d = max(rect[2], rect[3])
    #         region = cv2.resize(region, (d, d), interpolation=cv2.INTER_LINEAR)
    #     region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    #     mask_red1 = cv2.inRange(region_hsv, (0, 180, 50), (7, 255, 255))
    #     mask_red2 = cv2.inRange(region_hsv, (150, 180, 50), (180, 255, 255))

    #     imask_red1 = mask_red1>0
    #     imask_red2 = mask_red2>0
    #     region_red = np.zeros_like(region, np.uint8)
    #     region_red[imask_red1] = region[imask_red1]
    #     region_red[imask_red2] = region[imask_red2]

    #     region_red_gray = cv2.cvtColor(region_red, cv2.COLOR_BGR2GRAY)
    #     region_red_gray = cv2.medianBlur(region_red_gray, 5)

    #     circles = cv2.HoughCircles(region_red_gray, cv2.HOUGH_GRADIENT, 1, 3, param1=20, param2=int(region.shape[0]/8), minRadius=int(region.shape[0]/3), maxRadius=int(region.shape[0]*1.5))
    #     if circles is None:
    #         continue
    #     centered = False
    #     for x, y, r in circles[0, :]:
    #         if x > region.shape[1] / 3 and x < 2 * region.shape[1] / 3:
    #             if y > region.shape[0] / 3 and y < 2 * region.shape[0] / 3:
    #                 centered = True
    #                 break

    #     if not centered:
    #         continue

    #     lines = houghlines(region_red_gray, 0.4, 2)[0]
    #     horizontal_lines = 0
    #     for p, theta in lines:
    #         x0 = 0
    #         y0 = 0
    #         x1 = 0
    #         y1 = 0
    #         if sin(theta) != 0.0:
    #             x0 = 0
    #             y0 = p / sin(theta)
    #             x1 = region.shape[1]
    #             y1 = p / sin(theta) - (cos(theta)/sin(theta))*x1
    #         else:
    #             y0 = 0
    #             x0 = p / cos(theta)
    #             y1 = region.shape[1]
    #             x1 = p / cos(theta) - (sin(theta)/cos(theta))*y1
    #         if abs(y0 - y1) < region.shape[0] / 2:
    #             horizontal_lines += 1

    #     # print(horizontal_lines)
    #     if horizontal_lines >= 2:
    #         filtered.append(rect)

    # results = filtered
    total_guessed = len(results)

    for rect in results:
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 2)

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

    cv2.imwrite("result{}.jpg".format(i), frame)

    f1s.append(f1)
    TPRs.append(recall)

for f1 in f1s:
    print(f1)

print()
print(sum(f1s)/len(f1s))
print()

for TPR in TPRs:
    print(TPR)

print()
print(sum(TPRs)/len(TPRs))
print()
