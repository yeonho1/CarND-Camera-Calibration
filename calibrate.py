import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('./calibration_wide/GOPR????.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    if ret == True:
        objpoints.append(objp)

        # corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # imgpoints.append(corners2)
        imgpoints.append(corners)

        # img = cv2.drawChessboardCorners(img, (8, 6), corners2, ret)
        # img = cv2.drawChessboardCorners(img, (8, 6), corners, ret)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

img = cv2.imread('./calibration_wide/test_image2.png')

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img.shape[1::-1], None, None
)
undist = cv2.undistort(img, mtx, dist, None, mtx)

gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)


if ret == True:
    # cv2.line(undist, tuple(corners[40][0]), tuple(corners[0][0]), (0, 255, 0))
    src = np.float32([corners[0], corners[7], corners[47], corners[40]])
    dest = np.float32([[100, 100], [gray.shape[1] - 100, 100], [gray.shape[1] - 100, gray.shape[0] - 100], [100, gray.shape[0] - 100]])
    M = cv2.getPerspectiveTransform(src, dest)
    img = cv2.warpPerspective(gray, M, gray.shape[::-1], flags=cv2.INTER_LINEAR)
    cv2.imshow('warp', img)
    cv2.waitKey(0)

# cv2.imshow('undist', undist)
# cv2.waitKey(0)

cv2.destroyAllWindows()
