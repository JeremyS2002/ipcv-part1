# compute the best scaleFactor and minNeighbors for the cascade in NoEntry/cascake.xml measured on the images in No_entry

import numpy as np
import cv2
import argparse
from tqdm import tqdm

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

# optimized for best f1 score
best_f1 = 0
# the best f1 score achieved
bestf1Recall = 0
# the precision achieved with parameters optimizing for f1
bestf1Precision = 0
# the recall achieved with parameters optimizing for f1
bestRecall = 0
# the scale factor that gives the best f1
bestf1ScaleFactor = 0
# the min neighbors that gives the best f1
bestf1MinNeighbors = 0

# optimized for best recall
best_recall = 0
# the f1 achieved with parameters optimizing for recall
bestRecalF1 = 0
# the precision achieved with parameters optimizing for recall
bestRecalPrecision = 0
# the scale factor that gives the best recall
bestRecallScaleFactor = 0
# the min neigbors that gives the best recall
bestRecallMinNeighbors = 0

# optimized for best precision
best_precision = 0
# the f1 achieved with paramenters optimizing for precision
bestPrecisionF1 = 0
# the recall achieved with parameters optimizing for recall
bestPrecisionRecall = 0
# the scale factor that gives the best precision
bestPrecisionScaleFactor = 0
# the min neighbors that gives the best precision
bestPrecisionMinNeighbors = 0

scaleFactor = 1.01
for _ in tqdm(range(50)):
    for minNeighbors in range(5):
        ground_truth = 0
        found = 0
        total_guessed = 0

        for i in range(16):
            img_name = "../../No_entry/NoEntry{}.bmp".format(i)
            frame = cv2.imread(img_name)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)

            truth_name = "../../No_entry/NoEntry{}truth.txt".format(i)
            true = []
            with open(truth_name) as f:
                for line in f.readlines():
                    l = line.split(" ")
                    x = int(l[0])
                    y = int(l[1])
                    w = int(l[2])
                    h = int(l[3])
                    true.append((x, y, w, h))

            ground_truth += len(true)

            results = model.detectMultiScale(frame_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(8, 8), maxSize=(300, 300))
            total_guessed += len(results)
            
            for true_rect in true:
                detected = False
                for guess_rect in results:
                    if J(true_rect, guess_rect) > 0.3:
                        detected = True
                        break
                if detected:
                    found += 1            
                
        if total_guessed == 0:
            continue
        
        if found == 0:
            continue

        precision = found / total_guessed
        recall = found / ground_truth

        f1 = 2 * (precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
            bestf1Recall = recall
            bestf1Precision = precision
            bestf1ScaleFactor = scaleFactor
            bestf1MinNeighbors = minNeighbors
        
        if recall > best_recall:
            best_recall = recall
            bestRecalF1 = f1
            bestRecalPrecision = precision
            bestRecallScaleFactor = scaleFactor
            bestRecallMinNeighbors = minNeighbors

        if precision > best_precision:
            best_precision = precision
            bestPrecisionF1 = f1
            bestPrecisionRecall = recall
            bestPrecisionScaleFactor = scaleFactor
            bestPrecisionMinNeighbors = minNeighbors
    scaleFactor += 0.02
    

print(best_f1)
print(bestf1Recall)
print(bestf1Precision)
print(bestf1ScaleFactor)
print(bestf1MinNeighbors)
print()
print(bestRecalF1)
print(best_recall)
print(bestRecalPrecision)
print(bestRecallScaleFactor)
print(bestRecallMinNeighbors)
print()
print(bestPrecisionF1)
print(bestPrecisionRecall)
print(best_precision)
print(bestPrecisionScaleFactor)
print(bestPrecisionMinNeighbors)