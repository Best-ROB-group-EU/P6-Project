import numpy as np
import pyrealsense2 as rs
import cv2
import copy
import datetime
import open3d as o3d
from shapely.geometry import Polygon, Point
import time



class D435:
    def __init__(self):

        bag = r'/great_bag_path/477621_REAL.bag'

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
        for i in range(28):
            self.pipeline.wait_for_frames()
        while True:
            frames = self.pipeline.wait_for_frames()
            #Pre-processing, aligning frames, such that they match.
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            self.depth_frame = depth_frame
            self.color_frame = color_frame

            #Pixel values
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            #Intrinsic camera parameters (in this case color = depth, as the frames are aligned)
            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics


           # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)  #A colorized depth image of the area.


            self.depth_distance(58,405)
           # self.depth_distance(104,407)

            #color_cvt = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            self.show(color_image)

            break
    def depth_distance(self, x, y):
        depth_intrin = self.depth_intrin
        udist = self.depth_frame.get_distance(x, y)  # in meters
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], udist) #Pixels -> 3D point cloud projection
        point_array = np.asanyarray(point) #appending to a numpy array due to the formating of .ply

        #print(point)

        #--- For testing purposes ---#
        udist_test = self.depth_frame.get_distance(50, 74)
        test_point = [-0.4472, 0.39715,  1.235]
        
        #TODO: Cords [536, 96] IS IT MAPPED?


        test_point2 = [-0.16451279819011688, -0.09651362150907516, 0.3700000047683716]

        point_to_pixel = rs.rs2_project_point_to_pixel(depth_intrin, point)

        test_point_pixel = rs.rs2_project_point_to_pixel(depth_intrin, test_point)

        test_pixel = rs.rs2_deproject_pixel_to_point(depth_intrin, [630, 306], udist)

        #print(test_pixel)
        #print(udist_test)
        print(test_point_pixel)
        #print("DISTANCE: {0}".format(udist))
        # print(point_to_pixel)

        return point_array

    def rectangle_pointcloud(self, start_x1, start_y1, end_x1, end_y1):
        pointcloud = []

        #The end pixels can be smaller than a starting pixel:
        if start_x1 > end_x1:
            for x in range(start_x1, end_x1, -1):
                if start_y1 > end_y1:
                    for y in range(start_y1, end_y1, -1):
                        points = self.depth_distance(x, y)
                        pointcloud.append(points)
                else:
                    for y in range(start_y1, end_y1):
                        points = self.depth_distance(x, y)
                        pointcloud.append(points)
        else:
            for x in range(start_x1, end_x1):
                if start_y1 < end_y1:
                    for y in range(start_y1, end_y1):
                        points = self.depth_distance(x, y)
                        pointcloud.append(points)
                else:
                    for y in range(start_y1, end_y1, -1):
                        points = self.depth_distance(x, y)
                        pointcloud.append(points)

        self.img_origin[start_x1:end_x1, start_y1:end_y1] = (0, 0, 255)
        self.to_point_cloud_o3d(pointcloud)


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
        #print(minx, miny, maxx, maxy)

        #iterating through the pixels and converting them into 3D vectors
        box_patch = []
        for x in range(minx,maxx+1):
            for y in range(miny,maxy+1):
                box_patch.append([x,y])

        #Finding the points that belong to the polygon
        pixels = []
        for pb in box_patch:
            pt = Point(pb[0], pb[1])
            if(polygon.contains(pt)):
                pixels.append(self.depth_distance(int(pb[0]), int(pb[1])))
        #print(np.asarray(pixels))
        self.to_point_cloud_o3d(pixels)
        stop = time.time() - start_time
        print("Execution time: {}".format(stop))


    def to_point_cloud_o3d(self, points):
        path = "/home/vdr/Desktop/RealSense/great_pointcloud_path/" + str(datetime.datetime.now())
        pcd_poly = o3d.geometry.PointCloud()
        pcd_poly.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(path + "_REAL" + ".ply", pcd_poly)
        #o3d.visualization.draw_geometries([pcd_poly])


    def show(self, img):
        self.exterior = 0
        self.img_origin = img
        img_copy = copy.copy(self.img_origin)

        #self.polygon(133, 389, 140, 220, 248, 282, 229, 391)
        self.polygon(111, 293, 314, 292, 318, 440, 38, 444)
        #self.polygon(38,444, 54, 400, 155, 400, 150, 447)
        #self.polygon(104, 337, 406, 12, 630, 306, 238, 432)

        while True:
            if self.exterior != 0:
                cv2.fillPoly(self.img_origin, self.exterior, color=(255, 255, 0))
            cv2.imshow("Original color stream", img_copy)
            cv2.imshow("Color stream", self.img_origin)
            key = cv2.waitKey(10)

            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
class To_bag:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        # For real time capturing
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        self.depth_sensor = pipeline_profile.get_device().first_depth_sensor()

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        path = '/home/vdr/Desktop/RealSense/great_bag_path/' + str(datetime.datetime.now())
        self.config.enable_record_to_file(path + "_REAL" + '.bag')

        self.pipeline.start(self.config)

    def video(self):
        for i in range(50):
            self.pipeline.wait_for_frames()
            print("Frame: {}".format(i))
        self.pipeline.stop()
        print("Capture completed")

class To_JPG:
    def __init__(self):
        bag = r'/home/vdr/Desktop/RealSense/great_bag_path/2021-04-17 10:47:55.060879_REAL.bag'

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

        self.path = '/home/vdr/Desktop/RealSense/JPGs/'

        self.pipeline.start(config)
    def stream(self):
        try:
            align_to = rs.stream.color
            align = rs.align(align_to)
            i = 0
            while True:
                frames = self.pipeline.wait_for_frames()
                #Pre-processing, aligning frames, such that they match.
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                color_image = np.asanyarray(color_frame.get_data())

                cv2.imwrite(self.path + str(datetime.datetime.now()) + "_FRAMEID: " + str(i) + ".jpg", color_image)
                i += 1
        finally:
            pass
            print("Export completed")


if __name__ == "__main__":
    #To_bag().video()
    D435().video()
    #To_JPG().stream()




'''
#---- RIP IN PEACE ---


from: def video(self):
            #Manually aligning depth and color frames and visualizing them by using opencv
            
            
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape
    
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))
            
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RGB", images) #images
            cv2.waitKey(1)
            
            
            
from: def rectangle_pointcloud():
        for width in range(start_u1, end_u2):
            for heigth in range(start_v1, end_v2):
                points = self.depth_distance(width, heigth)
                pointcloud.append(points)
                
                
                
from: def show():
            #Projection check
        
            self.img_origin[129, 243] = (255, 0, 0)  #v, u
            self.img_origin[148, 199] = (255, 0, 0)
            self.img_origin[140, 296] = (255, 0, 0)
            self.img_origin[166, 241] = (255, 0, 0)
           # for y in range(241, 199, -1):
           #     print(y)
            update = cv2.cvtColor(self.img_origin, cv2.COLOR_BGR2RGB)
            cv2.imshow("Color stream", update)
            
            
            #Testing for the following values:
            #241 132      #u,v
            # [-0.04554886743426323, -0.056116145104169846, 0.3360000252723694]  #x,y,z
            
            
from :  def to_point_cloud_rs(self):
                pc = rs.pointcloud()
                point = pc.calculate(self.depth_frame)
                pc.map_to(self.color_frame)
                point.export_to_ply('new_test.ply', self.color_frame)

'''
