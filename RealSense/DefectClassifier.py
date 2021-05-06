import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
import time
import copy

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

        #bag = r'/home/vdr/Desktop/RealSense/great_bag_path/spot3.bag'
        bag = r'/home/frederik/Documents/Git/P6-Project/RealSense/great_bag_path/477621_REAL.bag'
        # bag = r'/home/vdr/Documents/spot_backyard_2021-03-25-09-48-00.bag'

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

    def video(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        for i in range(58):
            self.pipeline.wait_for_frames()
        while True:
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

            #Feed from YOLACT goes to fast_polygon
            self.fast_polygon(118, 294, 313, 293, 316, 441, 40, 445) # top left, right then buttom right, left
            #self.fast_polygon(328, 294, 521, 295, 607, 439, 334, 445)

            break
    def depth_distance(self, x, y):
        depth_intrin = self.depth_intrin
        udist = self.depth_frame.get_distance(x, y)  # in meters
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], udist) #Pixels -> 3D point cloud projection
        point_array = np.asanyarray(point) #appending to a numpy array due to the formating of .ply
        return point_array

    def fast_polygon(self, u1, v1, u2, v2, u3, v3, u4, v4):
        img_copy = copy.copy(self.color_image)
        new_img = np.zeros((480, 640))
        rectangle = np.array([[u1, v1], [u2, v2], [u3, v3], [u4, v4]], np.int32)
        start_time = time.time()
        cv2.drawContours(new_img, [rectangle], -1, 255, thickness=cv2.FILLED)

        yas = np.nonzero(new_img)
        stacked = np.dstack(yas)
        points = []

        for i in range(0, stacked.shape[1]):
            points.append(self.depth_distance(stacked[0][i][1], stacked[0][i][0]))

        self.to_point_cloud_o3d(points)
        diff = self.depth_fit_check(118, 294) - self.depth_fit_check(119, 288) #119, 288
        print("Depth difference: {0} cm".format(diff*100))
        stop = time.time() - start_time
        print("Execution time: {}".format(stop))

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
        errors = b - A * self.fit
        self.residual = np.linalg.norm(errors)
    def depth_fit_check(self, u, v):

        test = self.depth_distance(u, v)
        A = [test[0], test[1],1]
        z = np.squeeze(np.asarray(A* self.fit))

        New_A = np.asarray([A[0], A[1], z])

        #print("Difference: ", (test[2]-New_A[2]) * 100, " cm")
        return z

if __name__ == "__main__":
    D435_live().video()



'''
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
        '''