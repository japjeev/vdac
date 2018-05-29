#import the necessary packages
import cv2
import math

#Blob class
class blobz(object):

    def __init__(self, contour):

        global currentContour
        global currentBoundingRect
        global currentBoundingArea
        global centerPosition
        global centerPositions
        global cx
        global cy
        global dblCurrentDiagonalSize
        global dblCurrentAspectRatio
        global intCurrentRectArea
        global blnCurrentMatchFoundOrNewBlob
        global blnStillBeingTracked
        global intNumofConsecutiveFramesWithoutAMatch
        global predictedNextPosition
        global numPositions
        global blnBlobCrossedTheLine

        self.predictedNextPosition = []
        self.centerPosition = []
        currentBoundingRect = []
        currentContour = []
        self.centerPositions = []

        self.currentContour = contour
        self.currentBoundingArea = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        self.currentBoundingRect = [x, y, w, h]
        cx = (2*x + w)/2
        cy = (2*y + h)/2
        self.centerPosition = [cx, cy]
        self.dblCurrentDiagonalSize = math.sqrt(w*w + h*h)
        self.dblCurrentAspectRatio = (w/(h*1.0))
        self.intCurrentRectArea = w*h
        self.blnStillBeingTracked = True
        self.blnCurrentMatchFoundOrNewBlob = True
        self.blnBlobCrossedTheLine = False
        self.intNumofConsecutiveFramesWithoutAMatch = 0
        self.centerPositions.append(self.centerPosition)
    # predicted next position is weighted sum of last 5 positions
    def predictNextPosition(self):
        numPositions = len(self.centerPositions)
        if (numPositions == 1):
            self.predictedNextPosition = [self.centerPositions[-1][-2], self.centerPositions[-1][-1]]
        if (numPositions == 2):
            deltaX = self.centerPositions[1][0] - self.centerPositions[0][0]
            deltaY = self.centerPositions[1][1] - self.centerPositions[0][1]
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]
        if (numPositions == 3):
            SumofX = (self.centerPositions[2][0] - self.centerPositions[1][0])*2 + (self.centerPositions[1][0] - self.centerPositions[0][0])*1
            deltaX = (SumofX/3)
            SumofY = (self.centerPositions[2][1] - self.centerPositions[1][1])*2 + (self.centerPositions[1][1] - self.centerPositions[0][1])*1
            deltaY = (SumofY/3)
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]
        if (numPositions == 4):
            SumofX = (self.centerPositions[3][0] - self.centerPositions[2][0])*3 + (self.centerPositions[2][0] - self.centerPositions[1][0])*2 + (self.centerPositions[1][0] - self.centerPositions[0][0])*1
            deltaX = (SumofX/6)
            SumofY = (self.centerPositions[3][1] - self.centerPositions[2][1])*3 + (self.centerPositions[2][1] - self.centerPositions[1][1])*2 + (self.centerPositions[1][1] - self.centerPositions[0][1])*1
            deltaY = (SumofY/6)
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]
        if (numPositions >= 5):
            SumofX = (self.centerPositions[numPositions - 1][0] - self.centerPositions[numPositions - 2][0])*4 + (self.centerPositions[numPositions - 2][0] - self.centerPositions[numPositions - 3][0])*3 + (self.centerPositions[numPositions - 3][0] - self.centerPositions[numPositions - 4][0])*2 + (self.centerPositions[numPositions - 4][0] - self.centerPositions[numPositions - 5][0])*1
            deltaX = (SumofX/10)
            SumofY = (self.centerPositions[numPositions - 1][1] - self.centerPositions[numPositions - 2][1])*4 + (self.centerPositions[numPositions - 2][1] - self.centerPositions[numPositions - 3][1])*3 + (self.centerPositions[numPositions - 3][1] - self.centerPositions[numPositions - 4][1])*2 + (self.centerPositions[numPositions - 4][1] - self.centerPositions[numPositions - 5][1])*1
            deltaY = (SumofY/10)
            self.predictedNextPosition = [self.centerPositions[-1][-2] + deltaX, self.centerPositions[-1][-1] + deltaY]

##def PrintBlobzState(blobs):
##    for i in range(len(blobs)):
##        j = len(blobs[i].centerPositions)
##        print(str(i), str(j), str(blobs[i].predictedNextPosition), str(blobs[i].blnStillBeingTracked))
##        for k in range (0, j):
##            print(str(k), str(blobs[i].centerPositions[k][0]), str(blobs[i].centerPositions[k][1]))


def CheckIfBlobsCrossedTheLine(blobs, horizontalLinePosition):
    carCount = 0
    for existingBlob in blobs:
        if ((existingBlob.blnStillBeingTracked == True) and (len(existingBlob.centerPositions) >= 4) and (existingBlob.blnBlobCrossedTheLine == False)):
            if ((existingBlob.centerPositions[-1][-1] > horizontalLinePosition) and (existingBlob.centerPositions[-2][-1] <= horizontalLinePosition)):
                carCount += 1
                existingBlob.blnBlobCrossedTheLine = True
    return carCount

def matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs):
    for existingBlob in blobs:
        existingBlob.blnCurrentMatchFoundOrNewBlob = False
        existingBlob.predictNextPosition()
    for currentFrameBlob in currentFrameBlobs:
        intIndexOfLeastDistance = -1
        dblLeastDistance = 100000.0
        for i in range(len(blobs)):
            if (blobs[i].blnStillBeingTracked == True):
                dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions[-1],blobs[i].predictedNextPosition)
                if (dblDistance < dblLeastDistance):
                    dblLeastDistance = dblDistance
                    intIndexOfLeastDistance = i
        if (dblLeastDistance < (currentFrameBlob.dblCurrentDiagonalSize * 0.5)):
            blobs = addBlobToExistingBlobs(currentFrameBlob, blobs, intIndexOfLeastDistance)
        else:
            blobs, currentFrameBlob = addNewBlob(currentFrameBlob, blobs)
    for existingBlob in blobs:
        if (existingBlob.blnCurrentMatchFoundOrNewBlob == False):
            existingBlob.intNumofConsecutiveFramesWithoutAMatch = existingBlob.intNumofConsecutiveFramesWithoutAMatch + 1
        if (existingBlob.intNumofConsecutiveFramesWithoutAMatch >= 5):
            existingBlob.blnStillBeingTracked = False
    return blobs

def distanceBetweenPoints(pos1, pos2):
    if (pos2 == [] or pos2 == None):
        dblDistance = math.sqrt((pos1[0])**2 + (pos1[1])**2)
    else:
        dblDistance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
    return dblDistance

def addBlobToExistingBlobs(currentFrameBlob, blobs, intIndex):
    blobs[intIndex].currentContour = currentFrameBlob.currentContour
    blobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect
    blobs[intIndex].centerPositions.append(currentFrameBlob.centerPositions[-1])
    blobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize
    blobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio
    blobs[intIndex].blnStillBeingTracked = True
    blobs[intIndex].blnCurrentMatchFoundOrNewBlob = True
    return blobs

def addNewBlob(currentFrameBlob, blobs):
    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = True
    blobs.append(currentFrameBlob)
    return blobs, currentFrameBlob

def drawBlobInfoOnImage(blobs, m1):
    for i in range(len(blobs)):
        if (blobs[i].blnStillBeingTracked == True):
            x, y, w, h = blobs[i].currentBoundingRect
            cx = blobs[i].centerPositions[-1][-2]
            cy = blobs[i].centerPositions[-1][-1]
            cv2.rectangle(m1, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.circle(m1, (int(cx), int(cy)), 2, (0,0,0), -1)
            text = str(int(cx)) + "," + str(int(cy))
            cv2.putText(m1, text, (int(blobs[i].centerPositions[-1][-2]), int(blobs[i].centerPositions[-1][-1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return m1

##Globals
fcount = 0
frame = None
blobs = []
blnFirstFrame = True
total_carCount = 0
crossingLine = []

cam = cv2.VideoCapture('counting.mp4')

ret,frame = cam.read()
if ret is True:
    backSubtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
    run = True
else:
    run = False

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cam.get(3))
frame_height = int(cam.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('X','V','I','D'), 30, (frame_width,frame_height))

while(run):
    # Read a frame from the camera
    ret,frame = cam.read()

    # If the frame was properly read.
    if ret is True:
        fcount = fcount + 1

        # get the foreground
        foreGround = backSubtractor.apply(frame, None, 0.001)

        #### Filtering ####
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        # Fill any small holes
        closing = cv2.morphologyEx(foreGround, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)
        # threshold
        thresh = cv2.threshold(dilation, 15, 255, cv2.THRESH_BINARY)[1]

        (_, cnts, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)

        hulls = []

        for c in range(len(cnts)):
            hull = cv2.convexHull(cnts[c])
            hulls.append(hull)

        curFrameblobs = []
        l = 0
        for c in range(len(hulls)):
            l = l + 1
            ec = blobz(hulls[c])
            if (ec.intCurrentRectArea > 1600 and \
                ec.centerPosition[0] >= 250 and \
                ec.centerPosition[1] >= 400 and \
                fcount >= 5):
                curFrameblobs.append(ec)

        horizontalLinePosition = 0
        horizontalLinePosition, cols, _ = frame.shape
        horizontalLinePosition = horizontalLinePosition*0.55

        if (blnFirstFrame == True):
            crossingLine.append([0, horizontalLinePosition])
            crossingLine.append([cols - 1, horizontalLinePosition])
            for fl in curFrameblobs:
                blobs.append(fl)
        else:
            blobs = matchCurrentFrameBlobsToExistingBlobs(blobs, curFrameblobs)

        #PrintBlobzState(blobs)
        m1 = drawBlobInfoOnImage(blobs, frame)

        cv2.line(m1, (int(crossingLine[0][0]), int(crossingLine[0][1])), (int(crossingLine[1][0]), int(crossingLine[1][1])), (0, 0, 255))
        carCount = CheckIfBlobsCrossedTheLine(blobs, horizontalLinePosition)
        total_carCount = total_carCount + carCount
        cv2.putText(m1, str(fcount), (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.putText(m1, str(total_carCount), (150, 590), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        cv2.imshow('CCTV Feed',m1)
        cv2.imshow('CCTV Feed2',thresh)
        #cv2.imwrite("./frames/frame%d.jpg" % fcount, m1)
        # Write the frame into the file 'output.avi'
        out.write(m1)
        key = cv2.waitKey(10) & 0xFF
    else:
        break

    if key == 27:
        break

    blnFirstFrame = False

cam.release()
out.release()
cv2.destroyAllWindows()
