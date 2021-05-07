import cv2
import open3d as o3d
import numpy as np
import time

image = None
pcd = None


def show(new, ms=10000):
    # shows the image for testing purposes
    cv2.imshow("I wander if anyone will ever look at this...", new)
    cv2.waitKey(ms)
    cv2.destroyAllWindows()


def load(rgb="sidewalk.jpg", ply="sidewalk.ply"):
    # load the image and point cloud for testing purposes
    global image
    global pcd
    image = cv2.imread(rgb)
    pcd = o3d.io.read_point_cloud(ply)


def get_average(p1, p2, pointCloud):
    # receives the points to form a parallelogram and a points cloud
    # returns the average of all points in the point cloud that correspond
    # to the area of the parallelogram in the image
    x = 0.0
    y = 0.0
    z = 0.0
    n = abs(p2[0] - p1[0]) * abs(p2[1] - p1[1])
    for i in range(p1[0], p2[0] + 1):
        for j in range(p1[1], p2[1] + 1):
            point = pointCloud.points[(i + 1) * (j + 1)]
            x += point[0]
            y += point[1]
            z += point[2]
    x /= n ** 2
    y /= n ** 2
    z /= n ** 2
    return [x, y, z]


def calc(topLeftPoint, bottomRightPoint, pointCloud, img=None, visual=False):  # set visual to True, to show image
    # creates a ruler from the extremities of the side walk
    # returns the largest analyzed distance from it

    evaluation = 0
    side = bottomRightPoint[1] - topLeftPoint[1]  # side of each kernel
    n2 = int((bottomRightPoint[0] - topLeftPoint[0]) / (2 * side))  # number of pairs of kernels
    c = int(topLeftPoint[0] + (bottomRightPoint[0] - topLeftPoint[0]) / 2)  # center of the analyzed area
    mPoints = []
    if visual:
        color1 = (100, 100, 0)
        color2 = (150, 0, 50)
        color3 = (0, 100, 150)
        cv2.rectangle(img, (topLeftPoint[0], topLeftPoint[1]), (bottomRightPoint[0], bottomRightPoint[1]), color3, 3)
    # get points of kernels
    for i in range(n2):
        p1 = (c - side * (i + 1), topLeftPoint[1])  # tll
        p2 = (c - side * i, bottomRightPoint[1])  # brl
        p3 = (c + side * i, topLeftPoint[1])  # tlr
        p4 = (c + side * (i + 1), bottomRightPoint[1])  # brr
        if i == 0:
            if visual:
                cv2.rectangle(img, p1, p2, color3, -1)
                cv2.rectangle(img, p3, p4, color3, -1)
            else:
                pass
        else:
            # get z's
            mPoints.append(np.array(get_average(p1, p2, pointCloud)))
            mPoints.append(np.array(get_average(p3, p4, pointCloud)))
            if visual:
                cv2.rectangle(img, p1, p2, color1, 1)
                cv2.rectangle(img, p3, p4, color1, 1)
    el = get_average(topLeftPoint, (p4[0], bottomRightPoint[1]), pointCloud)
    er = get_average((p1[0], topLeftPoint[1]), bottomRightPoint, pointCloud)
    ruler = np.subtract(el, er)
    for point in mPoints:
        projection = np.dot(point, ruler) / np.sum(np.power(ruler, 2)) * ruler
        distance = np.linalg.norm(np.subtract(point, projection), 2)
        if distance > abs(evaluation):
            evaluation = abs(distance)
    if visual:
        cv2.rectangle(img, topLeftPoint, (p1[0], bottomRightPoint[1]), color2, -1)
        cv2.rectangle(img, (p4[0], topLeftPoint[1]), bottomRightPoint, color2, -1)
        show(img, 0)
    return evaluation


start = time.time()
load()
print("depression size:", calc((58, 410), (589, 440), pcd))
print("running time:", time.time() - start)