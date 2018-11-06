import numpy as np
import cv2
import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Size of the target pattern
boardSize = (7,6)

cbrow = boardSize[1]
cbcol = boardSize[0]

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Load in chessboard images
images = glob.glob('C:\Python27\cv\stamps\*.jpg')

for fname in images:

    img = cv2.imread(fname)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, boardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print(fname)
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, boardSize, corners,ret)
        cv2.imshow('img',img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Get the calibration parameters
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Print the parameters.
print('camera matrix: ' + str(mtx))
print('distortioin coeficients: ' + str(dist))
print('rotation vectors: ' + str(rvecs))
print('transformation vectors: ' + str(tvecs))

# Choose image to undistort 
img = cv2.imread('C:\Python27\cv\chessboard\WIN_20171028_22_01_32_Pro.jpg')
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# Undistort and write
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
     
cv2.imwrite('C:\Python27\cv\calibresult.png',dst)

# Find the average projection error
mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print "total error: ", mean_error/len(objpoints)