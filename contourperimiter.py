import cv2
import numpy as np
import glob


BLUR = 9
LOW_THRESH = 120
HIGH_THRESH = 255

LINE_COLOUR = (0,255,255)
LINE_WIDTH = 6

PERSPECTIVE_SIZE = 300

EPSILON_SCALE = 0.1

# Load stamps
images = glob.glob('C:\Python27\cv\stamp_only\*.jpg')

# Find corners of each stamp and perform perspective correction
for imageUrl in images:

	img = cv2.imread(imageUrl)

	# Read the image and convert to grayscale.
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# Blur the image to smooth edges.
	blurred = cv2.GaussianBlur(gray, (BLUR, BLUR), 0)

	# Threshold the image and invert the result.
	thresh = cv2.threshold(blurred, LOW_THRESH, HIGH_THRESH, cv2.THRESH_BINARY_INV)[1]

	# Find Contours
	im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)

	cv2.imshow('img2', thresh)

	# Find the largest contour.
	largestContour = 0
	largestContourIndex = 0
	for c in range(0, len(contours)):
		if(cv2.arcLength(contours[c],True) > largestContour):
			largestContour = cv2.arcLength(contours[c], True)
			largestContourIndex = c

	# Outer edge of stamp in training image.
	cnt = contours[largestContourIndex]

	# Get an approximate shape for the image.
	epsilon = EPSILON_SCALE * cv2.arcLength(cnt, True)
	approx = cv2.approxPolyDP(cnt, epsilon, True)

	# Draw the approximate shape.
	cv2.polylines(img, [approx], True, LINE_COLOUR, LINE_WIDTH)

	# Show the outer edge 
	cv2.imshow('img', img)

	# Sort the source points to match them to the destination points more easily
	approx = sorted(approx, key=lambda l:l[0][0])

	stampPoints = []

	#find top left then top right
	if approx[0][0][1] > approx[1][0][1]:
		stampPoints.append(approx[1][0])
		stampPoints.append(approx[0][0])
	else:
		stampPoints.append(approx[0][0])
		stampPoints.append(approx[1][0])

	#find bottom left then bottom right
	if approx[2][0][1] > approx[3][0][1]:
		stampPoints.append(approx[3][0])
		stampPoints.append(approx[2][0])
	else:
		stampPoints.append(approx[2][0])
		stampPoints.append(approx[3][0])

	#Source Points
	pts1 = np.float32(stampPoints)

	#Destination Points
	pts2 = np.float32([[0,0],[0, PERSPECTIVE_SIZE],[PERSPECTIVE_SIZE, 0],[PERSPECTIVE_SIZE, PERSPECTIVE_SIZE]])

	#Transformation Matrix
	M = cv2.getPerspectiveTransform(pts1,pts2)

	#Destination Image
	dst = cv2.warpPerspective(img,M,(PERSPECTIVE_SIZE, PERSPECTIVE_SIZE))

	plt.subplot(121),plt.imshow(img),plt.title('Input')
	plt.subplot(122),plt.imshow(dst),plt.title('Output')
	plt.show()

	cv2.waitKey(0)
	cv2.destroyAllWindows()
