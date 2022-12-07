import cv2

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

cascade_name = "../../NoEntrycascade/cascade.xml"
model = cv2.CascadeClassifier(cascade_name)

f1s = []
TPRs = []
Recalls = []

for i in range(16):
    # open and preprocess the image
    img_name = "../../No_entry/NoEntry{}.bmp".format(i)
    frame = cv2.imread(img_name)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # open and parse the ground truth file
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

    # number of stop signs actually present in the image
    ground_truth = len(true)

    # detect stop signs
    results = model.detectMultiScale(frame_gray, scaleFactor=1.07, minNeighbors=3, minSize=(8, 8), maxSize=(300, 300))
    # results = model.detectMultiScale(frame_gray, scaleFactor=1.01, minNeighbors=1, minSize=(8, 8), maxSize=(300, 300))
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

    # save 3 output images for report
    output_name = "result{}.jpg".format(i)
    cv2.imwrite(output_name, frame)


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

for recall in Recalls:
    print(recall)

print()
print(sum(Recalls)/len(Recalls))
print()