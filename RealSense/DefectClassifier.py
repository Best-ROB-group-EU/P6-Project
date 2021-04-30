import numpy as np
import pyrealsense2 as rs
import cv2
import open3d as o3d
from shapely.geometry import Polygon, Point
import time

class D435_live:
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
    def video(self):
        align_to = rs.stream.color
        align = rs.align(align_to)
        for i in range(10):
            self.pipeline.wait_for_frames()
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            self.depth_frame = depth_frame
            self.color_frame = color_frame

            #Pixel values
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            self.polygon(118, 294, 313, 293, 316, 441, 40, 445) #change these ones continuosly

            '''
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Color stream', color_image)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('q') or key == 27:
                #print(color_image.shape)
                #print(depth_image.shape)
                cv2.destroyAllWindows()
                break
            '''
    def depth_distance(self, x, y):
        depth_intrin = self.depth_intrin
        udist = self.depth_frame.get_distance(x, y)  # in meters
        point = rs.rs2_deproject_pixel_to_point(depth_intrin, [x, y], udist) #Pixels -> 3D point cloud projection
        point_array = np.asanyarray(point) #appending to a numpy array due to the formating of .ply
        return point_array
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
        box_patch = []
        for x in range(minx,maxx+1):
            for y in range(miny,maxy+1):
                box_patch.append([x,y])

        #Finding the points that belong to the polygon
        pixels = []
        #pixels_poly =[]
        for pb in box_patch:
            pt = Point(pb[0], pb[1])
            if(polygon.contains(pt)):
                #pixels_poly.append([int(pb[0]), int(pb[1])])
                pixels.append(self.depth_distance(int(pb[0]), int(pb[1])))
        #print(np.asarray(pixels))
        self.to_point_cloud_o3d(pixels)

        self.depth_fit_check(u1, v1)
        self.depth_fit_check(u2, v2)
        self.depth_fit_check(u3, v3)
        self.depth_fit_check(u4, v4)
        stop = time.time() - start_time
        print("Execution time: {}".format(stop))
    def to_point_cloud_o3d(self, points):
        pcd_poly = o3d.geometry.PointCloud()
        pcd_poly.points = o3d.utility.Vector3dVector(points)
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
    def depth_fit_check(self, u, v): #4 corners instead of 1 needed.
        #print("Corner pixels u: {0}, v: {1}:".format(u, v))
        test = self.depth_distance(u, v)
        A = [test[0], test[1],1]
        z = np.squeeze(np.asarray(A* self.fit))

        New_A = np.asarray([A[0], A[1], z])
        #print("Raw depth data:", test)
        #print("Fit depth data:", New_A)
        #print("Difference: ", test[2]-New_A[2])
        return z

if __name__ == "__main__":
    D435_live().video()