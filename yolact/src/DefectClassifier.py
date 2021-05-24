#!/usr/bin/env python
import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import time
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import rospy
#from std_msgs.msg import String
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge
import math

transform = np.array([[0,   0.5736, -0.8192, -0.0500],
                      [1,   0,       0,       0     ],
                      [0,  -0.8192, -0.5736,  0.9800],
                      [0,   0,       0,       1.0000]])

tile1 = []


class D435_live:
    def __init__(self):
        '''
        #----------LIVE-----------#
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        '''
        #----------BAG-----------#
        bag = r'/home/vdr/Desktop/RealSense/good_data_test/EDGES_2.5_CM.bag'

        self.pipeline = rs.pipeline()
        config = rs.config()

        # From a bag file
        config.enable_device_from_file(bag, False)
        config.enable_all_streams()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        playback = device.as_playback()
        playback.set_real_time(False)

        self.pipeline.start(config)
        #self.pub = rospy.Publisher('image', Image, queue_size=1)
        #rospy.init_node('D435_Image_Publisher', anonymous=True, disable_signals =True)
        #self.rate = rospy.Rate(1)  # 10hz
        #self.sub = rospy.Subscriber('image_publisher', Image, self.callback)

    '''
    def img_publisher(self, image):
        bridge = CvBridge()
        imgMsg = bridge.cv2_to_imgmsg(image, "bgr8")
        #rospy.loginfo(imgMsg)
        self.pub.publish(imgMsg)
    '''

    '''
    def callback(self, data):
        current_frame = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        cv2.imshow("Subscriber", current_frame)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            rospy.signal_shutdown("shutdown")
        #TODO: Get corner msg, and pass it to self.fast_polygon(argument) function. There do the conversion to np.array if needed
        #Now it runs as long as there are available frames
    '''
    #def edge_detection(self):


    def video(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        for i in range(30):
            self.pipeline.wait_for_frames()
        while True: #and not rospy.is_shutdown():
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            self.depth_frame = depth_frame
            self.color_frame = color_frame

            #Pixel values
            self.color_image = np.asanyarray(color_frame.get_data())

            depth_image = np.asanyarray(depth_frame.get_data())

            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            #self.img_publisher(self.color_image)
            #self.rate.sleep()

            #Feed from YOLACT goes to fast_polygon:

            #Image 1267, from 20210507_150607.bag
            #tiles = [[1, [[300, 203], [61, 216], [0, 308], [0, 479], [275, 479]]], [2, [[639, 216], [563, 117], [372, 123], [392, 280], [639, 280]]], [3, [[389, 280], [412, 479], [639, 479], [639, 282]]], [4, [[68, 207], [299, 200], [307, 98], [151, 84]]]]
            tiles = [[1, [[355, 303], [352, 201], [506, 198], [550, 301]]], [2, [[356, 313],[551, 314], [615, 478], [348, 479]]]]
            ruler_region = [[19, 380], [639, 380], [639, 350], [19, 350]]


            ###################################
            start = time.time()
            edge_defects = self.edge_detector(tiles)
            print("Tile check:", time.time()-start, "s")
            for i in range(len(edge_defects)):
                if not "Low severity" in edge_defects[i]:
                    print(edge_defects[i][2], "edge defect detected at", edge_defects[i][0], "with size:", edge_defects[i][1],"cm")
                    #Save image with tags, maybe draw circle in current image at this pos

            start = time.time()
            depression_defects = self.depression_detector(ruler_region)
            print("Depression check:", time.time()-start, "s")
            for i in range(len(depression_defects)):
                if not "Low severity" in depression_defects[i]:
                    print(depression_defects[i][2], "depression defect detected at", depression_defects[i][0], "with size:", depression_defects[i][1],"cm")
                    #Save image with tags, maybe draw circle in current image at this pos
            ###################################
            '''
            while True:
                cv2.imshow("Original stream", self.color_image)
                key = cv2.waitKey(10)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    break
            '''
            #print(tile1)
            fig = plt.figure(figsize=plt.figaspect(0.5))
            #fig.ylim([-1, 1])
            #fig.xlim([-1, 1])
            #ax = plt.subplot(111, projection='3d')
            ax = fig.add_subplot(1,2,1, projection='3d')
            # plt.subplot(1,2,1)
            ax.scatter(tile1[0][0], tile1[0][1], tile1[0][2], color='b')
            ax.set_zlim([0, 0.5])
            ax = fig.add_subplot(1,2,2, projection ='3d')
            ax.scatter(tile1[1][0], tile1[1][1], tile1[1][2], color='g')
            ax.set_zlim([0, 0.5])
            plt.show()


            break


    def depth_distance(self, x, y):
        depth_intrin = self.depth_intrin
        udist = self.depth_frame.get_distance(x, y)  # in meters
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], udist) #Pixels -> 3D point cloud projection
        point_array = np.asanyarray(point) #appending to a numpy array due to the formating of .ply
        return point_array


    def edge_detector(self, list):
        edges = []
        # Make list of corner pixels with heights
        corner_heights = []
        for i in range(len(list)):
            current_set = list[i]
            corners = current_set[1]

            rectangle = np.asarray(corners).astype("int32")

            new_img = np.zeros((480, 640))
            cv2.drawContours(new_img, [rectangle], -1, 255, thickness=cv2.FILLED)

            yas = np.nonzero(new_img)
            stacked = np.dstack(yas)
            points = [] # 3D points inside polygon

            for j in range(0, stacked.shape[1], int(stacked.shape[1]/2000)):
                point3d = self.depth_distance(stacked[0][j][1], stacked[0][j][0])
                point3d = np.append(point3d, 1)
                t_point = np.matmul(transform,point3d)
                #points.append([point3d[0], point3d[1], point3d[2]])
                points.append([t_point[0], t_point[1], t_point[2]])

            self.to_point_cloud_o3d(points)

            for j in corners:
                height = float(self.depth_fit_check(j[0],j[1]))
                corner_heights.append([j, height])

        # Check corner height list for close neighbors, and then check height difference
        threshold = 15 #pixels
        edges = []
        for i in range(len(corner_heights)):
            c_point = corner_heights[i][0]
            for j in range(i+1, len(corner_heights)):
                dist = self.euclidean_distance(c_point, corner_heights[j][0])
                if dist < threshold:
                    diff = abs(corner_heights[i][1] - corner_heights[j][1])
                    diff_cm = diff*100
                    if diff_cm < 1:
                        #Either pass because this is very low or store as severity 1
                        edges.append([c_point, round(diff_cm,3), "Low severity"])
                    elif diff_cm > 1 and diff_cm < 3:
                        edges.append([c_point, round(diff_cm,3), "Medium severity"])
                    elif diff_cm > 3:
                        edges.append([c_point, round(diff_cm,3), "High severity"])
        return edges


    def get_average_height(self, p1, p2):
        # receives the points to form a parallelogram and a points cloud
        # returns the average of all points in the point cloud that correspond
        # to the area of the parallelogram in the image
        z = 0.0
        n = abs(p2[0] - p1[0]) * abs(p2[1] - p1[1])
        for i in range(p1[0], p2[0] + 1):
            for j in range(p1[1], p2[1] + 1):
                point3d = self.depth_distance(i, j)
                point3d = np.append(point3d, 1)
                t_point = transform.dot(point3d)
                z += t_point[2]
        z = z / n
        return z


    def depression_detector(self, list):
        # creates a ruler from the extremities of the side walk
        # returns the largest analyzed distance from it

        # Fit plane to edges
        left_points = np.array([list[0], [list[0][0]+30, list[0][1]], [list[3][0]+30, list[3][1]], list[3]]) #ll, ul, ll + 30 on x, ul + 30 on x
        right_points = np.array([list[1], [list[1][0]-30, list[1][1]], [list[2][0]-30, list[2][1]], list[2]]) #lr, ur, lr - 30 on x, ur - 30 on x
        new_img = np.zeros((480, 640))
        cv2.drawContours(new_img, [left_points, right_points], -1, 255, thickness=cv2.FILLED)

        yas = np.nonzero(new_img)
        stacked = np.dstack(yas)
        points = [] # 3D points inside polygon

        for i in range(stacked.shape[1]):
            point3d = self.depth_distance(stacked[0][i][1], stacked[0][i][0])
            point3d = np.append(point3d, 1)
            t_point = transform.dot(point3d)
            points.append([t_point[0], t_point[1], t_point[2]])

        self.to_point_cloud_o3d(points)

        evaluation = 0
        kernel_size = list[0][1] - list[3][1]  # Size of kernel, y1-y2

        #n2 = int((topRightPoint[0] - topLeftPoint[0]) / (2 * kernel_size))  # x1-x2, x span between sides
        cx = int((list[3][0] + list[2][0]) / 2)  # center of the analyzed area, WHERE THE COBBLESTONE MIGHT BE
        cy = int((list[3][1] + list[1][1]) / 2) # CENTER LINE

        left_kernels = np.arange(cx-50, list[3][0], -30)
        right_kernels = np.arange(cx+50, list[2][0], 30)
        kernels = np.concatenate((left_kernels, right_kernels), axis=0)
        depressions = []
        largest_left = 0
        for x in kernels:
            center_point = [x, cy]
            tl = [x-15, cy-15]
            br = [x+15, cy+15]
            height = self.get_average_height(tl,br)
            p_height = self.depth_fit_check(x, cy)
            diff = (p_height-height)*100
            if diff < 2:
                depressions.append([center_point, round(diff,3), "Low severity"])
            elif diff < 4 and diff > 2:
                depressions.append([center_point, round(diff,3), "Medium severity"])
            elif diff > 4:
                depressions.append([center_point, round(diff,3), "High severity"])
        return depressions


    def euclidean_distance(self, point1, point2):
        r = math.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
        return r

    def fast_polygon(self):
       # 118, 294, 313, 293, 316, 441, 40, 445)
        rectangle_a = []
        mylist = ["1", [["118", "294"], ["313", "293"], ["316", "441"], ["40", "445"]]]

        #mylist[0] for mask index

        modified_list = [list(map(int, i)) for i in list(mylist[1])]
        for i in modified_list:
            rectangle_a.append(i)

        rectangle_a = [[300, 203], [61, 216], [0, 308], [0, 479], [275, 479]]

        rectangle = np.asarray(rectangle_a).astype("int32")
        img_copy = copy.copy(self.color_image)
        new_img = np.zeros((480, 640))
        start_time = time.time()
        cv2.drawContours(new_img, [rectangle], -1, 255, thickness=cv2.FILLED)

        yas = np.nonzero(new_img)
        stacked = np.dstack(yas)
        points = [] # 3D points inside polygon

        for i in range(0, stacked.shape[1]):
            points.append(self.depth_distance(stacked[0][i][1], stacked[0][i][0]))

        self.to_point_cloud_o3d(points)

        diff = self.depth_fit_check(328, 294) - self.depth_fit_check(313, 293) #119, 288

        """
        while True:
            #cv2.drawContours(new_img, [rectangle], -1, 255, thickness=cv2.FILLED)
            cv2.drawContours(img_copy, [rectangle], -1, 255, thickness=cv2.FILLED)
            cv2.imshow("Original stream", self.color_image)
            cv2.imshow("Polygon", img_copy)
            key = cv2.waitKey(10)
            #cv2.imshow("haha", new_img)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
        """

    def to_point_cloud_o3d(self, points):
        pcd_poly = o3d.geometry.PointCloud()
        pcd_poly.points = o3d.utility.Vector3dVector(points)
        #o3d.visualization.draw_geometries([pcd_poly])
        self.linear_fit(pcd_poly)



    def linear_fit(self, pcd_poly):

        point_cloud = pcd_poly

        downpcd = point_cloud.voxel_down_sample(voxel_size = 0.02)

        pcl_np = np.asarray(downpcd.points)

        x = pcl_np[:, 0]
        y = pcl_np[:, 1]
        z = pcl_np[:, 2]

        tmp_A = []
        tmp_b = []

        for i in range(len(x)):
            tmp_A.append([x[i], y[i], 1])
            tmp_b.append(z[i])
        b = np.matrix(tmp_b).T
        A = np.matrix(tmp_A)

        self.fit = (A.T * A).I * A.T * b
        Fit_linear = A*self.fit
        errors = b - A * self.fit
        #self.residual = np.linalg.norm(errors)
        tile1.append([x, y, Fit_linear])

        #fig = plt.figure(figsize=plt.figaspect(0.5))
        '''
        plt.figure()
        plt.ylim([-1, 1])
        plt.xlim([-1, 1])
        ax = plt.subplot(111, projection ='3d')
        #ax = fig.add_subplot(1,2,1, projection='3d')
        #plt.subplot(1,2,1)
        ax.scatter(x, y, z, color ='b')
        ax.set_zlim([-1, 1])
        #ax = fig.add_subplot(1,2,2, projection ='3d')
        ax.scatter(x, y, Fit_linear, color ='g')
        ax.set_zlim([-1, 1])
        plt.show()
        '''



    def depth_fit_check(self, u, v):
        test = self.depth_distance(u, v)
        A = [test[0], test[1],1]
        z = np.squeeze(np.asarray(A* self.fit))
        return z

if __name__ == "__main__":
    D435_live().video()
    #read from bag for image and depth
    ##store image as jpg for yolact

    #read yolact points


    #for all tiles fast_polygon ()
        #for i in polygons linear_fit(fast_polygon)
            #for all corners do depth_fit_check
            #save corner heights to list



    #polygons for each tile
    ##fit plane to each tile            fit_plane(points)
    ###check corners height on planes           depth_fit_check(corner points)
    ####for loop for each tiles corners and calculate cartesian distance for close points (distances between u's and v's)
    #####if distance is less than 10 pixels, check difference
    ######check severity level
    #######if high severity, mark image somehow



    #


    '''
    try:
        D435_live().video()
    except rospy.ROSInterruptException:
        pass
    '''
    """

def polygon(self, u1, v1, u2, v2, u3, v3, u4, v4): # 4 points: [u1, v1], [u2, v2], [u3,v3], [u4,v4]

        start_time = time.time()

        #Stored them in a list just to keep an easy track of it
        x = [u1-1, u2, u3+1, u4]
        y = [v1, v2-1, v3, v4+1]

        polygon = Polygon([(x[0], y[0]), (x[1], y[1]), (x[2], y[2]), (x[3], y[3])])

        #for plotting purposes
        int_coords = lambda x: np.array(x).round().astype(np.int32)
        self.exterior = [int_coords(polygon.exterior.coords)]

        #getting the boundaries of a polygon
        minx, miny, maxx, maxy = polygon.bounds
        minx, miny, maxx, maxy = int(minx), int(miny), int(maxx), int(maxy)

        #iterating through the pixels and converting them into 3D vectors
        #box_patch = np.empty((maxx-minx, maxy-miny))
        box_patch = []
        for x in range(minx,maxx+1):
            for y in range(miny,maxy+1):
                #box_patch = np.append(box_patch,[x,y])
                box_patch.append([x,y])
        print("Width: {0}, Heigth: {1}".format((maxx-minx), (maxy-miny)))

        #Finding the points that belong to the polygon
        pixels = []
        pixels_poly =[]
        width = (maxx-minx)
        height = (maxy - miny)
        # TODO: Start from here

        for i in range(0, width):  # prev: 0 to width
            for j in range(0, height):  # prev 0 to height
                pt = Point(box_patch[height * j + i + j])
                if (polygon.contains(pt)):
                    for u in range(height, 0, -1):
                        ptu = Point(box_patch[height * u + i + u])
                        if (polygon.contains(ptu)):
                            pixels += box_patch[(height * j + i + j): (height * u + i + u): height + 1]
                            #pixels += box_patch[(height * j * i + j):: height +1]
                            break
                    break


        print("min: {0}, max: {1}".format(min(pixels), max(pixels)))


                    #for u in range(height, 0):
                     #   ptu = Point(box_patch[height*u+i+u])
                      #  print(u)
                        #if(polygon.contains(ptu)):
                         #   for k in range(j, u):
                          #      pixels.append(self.depth_distance(int(k), int(i)))
                            #break
                    #break


        for pb in box_patch:
            pt = Point(pb[0], pb[1])
            if(polygon.contains(pt)):
                pixels_poly.append([int(pb[0]), int(pb[1])])
                pixels.append(self.depth_distance(int(pb[0]), int(pb[1])))



        stop = time.time() - start_time
        #print(np.asarray(pixels))
        print("Execution time: {}".format(stop))

        img_origin = self.color_image
        img_copy = copy.copy(img_origin)
        while True:
            if self.exterior != 0:
                cv2.fillPoly(img_origin, self.exterior, color=(255, 255, 0))
            cv2.imshow("Original color stream", img_copy)
            cv2.imshow("Color stream", img_origin)
            key = cv2.waitKey(10)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break


        self.to_point_cloud_o3d(pixels)

        self.to_point_cloud_o3d(pixels)

        self.depth_fit_check(u1, v1)
        self.depth_fit_check(u2, v2)
        self.depth_fit_check(u3, v3)
        self.depth_fit_check(u4, v4)
        #stop = time.time() - start_time
        #print("Execution time: {}".format(stop))
"""
