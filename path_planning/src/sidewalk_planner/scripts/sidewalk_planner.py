import copy
import cv2
import rospy

import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Path
from shapely.geometry import Polygon
from shapely.affinity import affine_transform
from shapely.ops import unary_union
from descartes.patch import PolygonPatch

# =================== TEST DATA ===================
# gods of code, forgive me
sidewalk_xyz = [
                [[-0.5321995102478504, -0.41964998841285706, 0.10618141478346277],
                 [-2.886153137049627, -0.39940157532691956, 0.023984670175945674],
                 [-2.744444110101366, 0.3892264664173126, 0.036385390580021354],
                 [-0.5224942166309238, 0.42163586616516113, 0.11445887927434531]],
                [[-0.5338384779075265, -0.4961068034172058, 0.10834945562653985],
                 [-1.7252574782967567, -1.0032498836517334, 0.03793246742350888],
                 [-3.1143489551057817, -0.9654291868209839, -0.03830431833527673],
                 [-2.828329564472914, 1.0819342136383057, -0.0034526448121308395],
                 [-2.0554307251613855, 1.0957368612289429, 0.03929073183729277],
                 [-0.5189747752098323, 0.46524325013160706, 0.11812886696723748]],
                [[-0.5230751141008854, -0.48507052659988403, 0.111686882451389],
                 [-1.5696194292607366, -0.9344045519828796, 0.04463284272768865],
                 [-3.0850397580247875, -0.8590090274810791, -0.04616624466084465],
                 [-2.892001712445879, 1.1431313753128052, -0.017152857357758317],
                 [-2.3207915847876786, 1.2004801034927368, 0.017816040930678734],
                 [-0.5184622394742608, 0.4647347927093506, 0.12020259798311891]],
                [[-0.5338384779075265, -0.4961068034172058, 0.07069543810382112],
                 [-0.9994692667726784, -0.7036938667297363, 0.0278212777354519],
                 [-3.4296777254588604, 0.017446212470531464, -0.12800394146838928],
                 [-2.865724254997873, 1.416576623916626, -0.03286500098313916],
                 [-0.5215374803909302, 0.4677855670452118, 0.10791233601485553]],
                [[-0.5574153676196575, -0.5202814936637878, 0.04957367762246881],
                 [-1.0632871545127571, -0.7509921193122864, -0.02609741363589113],
                 [-4.136135639610576, 0.021093012765049934, -0.4093661359870106],
                 [-3.7055904628924368, 1.8391090631484985, -0.2879717902288546],
                 [-0.5415265934508204, 0.4876156151294708, 0.0871169278100288]]
                ]

yolact_xyz = [
              [[-2.931417821991634, -0.40955430269241333, 0.022304623410251925],
               [-0.5261503550175548, -0.41155725717544556, 0.10646949103954152],
               [-0.5210249446553588, 0.4167163074016571, 0.11446388593349763],
               [-2.744444110101366, 0.38491249084472656, 0.03634468694069817]],
              [[-2.948743619984674, -0.9763625264167786, -0.029275362989610976],
               [-0.533325915668714, -0.48032206296920776, 0.10852483180625636],
               [-0.5184622394742608, 0.4573398530483246, 0.11808346577517624],
               [-2.828329564472914, 0.9395885467529297, -0.004779321262284364]],
              [[-2.5040184261524914, -1.2350213527679443, -0.014363649521535904],
               [-0.5230751141008854, -0.4507187306880951, 0.11198480382587574],
               [-0.5174371871596812, 0.43567854166030884, 0.1200124631724468],
               [-2.658076406496906, 1.0060733556747437, -0.004224531502937556]],
              [[-3.293334404801654, -1.7111479043960571, -0.1828202111042463],
               [-2.043809831954801, -0.325825035572052, -0.03658281959871715],
               [-0.5642592781546354, -0.5087196826934814, 0.0679301051894702],
               [-0.5551522402390838, 0.4472115933895111, 0.10460678547602505],
               [-2.764400091923022, 1.2753239870071411, -0.030554247421916342]],
              [[-2.902859302308464 -1.50513756275177, -0.2982898820416094],
               [-2.2291959778975965, -0.3561205267906189, -0.16793629920681497],
               [-0.5876347136773705, -0.5318433046340942, 0.04513132670221673],
               [-0.5727181178234935, 0.46276265382766724, 0.08207763546152186],
               [-3.36306233717432, 1.5565972328186035, -0.25215301419465924]]
              ]

# ================= TEST DATA END =================

class Pose:
    """
    Class for holding a robot pose, used for the plan
    """
    def __init__(self, position, orientation):
        self.x = position[0]
        self.y = position[1]
        self.z = 0
        # Quaternion conversion from axis-angle
        self.theta = orientation[2]
        self.q = Rotation.from_rotvec(orientation).as_quat()

    def to_ros_msg(self, frame="map"):
        """
        Convert Pose to ROS PoseStamped message

        :param frame: Name of the ROS tf-frame
        :return:      ROS Pose (geometry msg)
        """
        msg = PoseStamped()
        msg.pose.position = Point(self.x, self.y, self.z)
        msg.pose.orientation = Quaternion(self.q[0], self.q[1], self.q[2], self.q[3])
        msg.header.frame_id = frame
        msg.header.stamp = rospy.Time.now()
        return msg


class Plan:
    """
    Class for holding and manipulating a planned path
    """
    def __init__(self):
        self.poses = list()

    def add_pose(self, pose):
        """
        Appends a new via-point to the path

        :param pose: Pose
        :return:
        """
        self.poses.append(pose)

    def clear_plan(self):
        """
        Clears the current plan

        :return:
        """
        self.poses = []

    def to_ros_msg(self, frame="map"):
        """
        Convert Plan to ROS Path message

        :param frame: Name of the ROS tf-frame
        :return:      ROS Path (nav msg)
        """
        msg = Path()
        for pose in self.poses:
            msg.poses.append(pose.to_ros_msg(frame))

        msg.header.frame_id = frame
        msg.header.stamp = rospy.Time.now()
        return msg

    def from_ros_msg(self, msg):
        """
        Create Plan given ROS Path msg

        :param msg: The path message
        :return:
        """
        for point in msg.poses:
            position = [point.pose.position.x, point.pose.position.y, point.pose.position.z]
            orientation = Rotation.from_quat([point.pose.orientation.x,
                                              point.pose.orientation.y,
                                              point.pose.orientation.z,
                                              point.pose.orientation.w]).as_rotvec()
            self.add_pose(Pose(position, orientation))


class Node:
    """
    Used for constructing an undirected, unweighted graph with adjacency list
    """
    def __init__(self, id, children=None):
        self.id = id
        self.position = np.zeros((1, 2))
        if children is None:
            self.children = list()
        else:
            self.children = children

        # Extra attributes for A* search
        self.h = 0
        self.g = 0
        self.f = 0
        self.backpointer = self

    def __lt__(self, other):
        # Defines behaviour for the "less than" comparison operator
        # Node1 < Node2 is equivalent to Node1.f < Node2.f
        # Enables the use sort() in the A* planner
        return self.f < other.f

    def __eq__(self, other):
        # Defines behaviour for equality comparison operator
        # Node1 == Node2 is equivalent to Node1.id == Node2.id
        return self.id == other.id

    def __ne__(self, other):
        # Defines behaviour for inequality comparison operator
        # Node1 != Node2 is equivalent to Node1.id != Node2.id
        return self.id != other.id

    def add_child(self, child_node):
        """
        Adds a child to the node

        :param child_node: ID of the node to add as child
        :return:
        """
        if child_node not in self.children:
            self.children.append(child_node)
            self.children.sort()
            return True
        else:
            return False


class Graph:
    """
    Class for undirected, unweighted graphs
    """
    def __init__(self):
        self.nodes = dict()

    def add_node(self, node):
        """
        Adds a node to the graph

        :rtype:      bool
        :param node: instance of object type Node to add to Graph
        :return:     True if node was successfully added, otherwise False
        """
        if isinstance(node, Node) and node.id not in self.nodes:
            self.nodes[node.id] = node
            return True
        else:
            return False

    def add_edge(self, n1, n2):
        """
        Adds edge between 2 nodes n1 and n2

        :rtype:    bool
        :param n1: First node
        :param n2: Second node
        :return:   True if successful addition, otherwise False
        """
        if n1 in self.nodes and n2 in self.nodes:
            for k, v in self.nodes.items():
                if k == n1:
                    v.add_child(n2)
                if k == n2:
                    v.add_child(n1)

            return True
        else:
            return False


class SidewalkPolygon:
    def __init__(self, grid_dimensions=(480, 640, 3)):
        self.vertices = list()
        self.hull = np.asarray([0, 0])
        self.sidewalk = np.zeros(grid_dimensions)
        self.poly = Polygon()
        self.sidewalk_indices = None

    def set_vertices(self, v):
        self.vertices = v
        # OpenCV data type error fix
        if self.vertices.dtype == 'float64':
            self.vertices = np.float32(self.vertices)

    def compute_sidewalk_outline(self):
        self.hull = cv2.convexHull(self.vertices)
        self.hull = np.reshape(self.hull, (self.hull.shape[0], 2))

    def fill_sidewalk(self, use_hull=True):
        """
        For drawings in the image space [u,v]
        :param use_hull:
        :return:
        """
        self.sidewalk = np.zeros(self.sidewalk.shape)
        # Datatype fix for cv and shapely compatibility/indifference
        pts = self.vertices
        if use_hull:
            pts = self.hull

        if self.vertices.dtype == 'float32':
            self.poly = Polygon(pts)
        else:
            cv2.drawContours(self.sidewalk, [pts], -1, (255, 255, 255), thickness=cv2.FILLED)

    def compute_sidewalk_indices(self):
        self.sidewalk_indices = np.vstack(np.nonzero(self.sidewalk))

    def is_in_sidewalk(self, pt, use_hull=True):
        """
        Method for checking if a point is inside the sidewalk

        :param pt:       Point to check geometry for
        :param use_hull: Use convex hull as sidewalk polygon
        :return:
        """
        pixel = (int(round(pt[0], 0)), int(round(pt[1], 0)))
        if self.vertices.dtype == 'float32':
            p = shapely.geometry.Point(pt)
            return self.poly.contains(p)
        elif use_hull:
            if cv2.pointPolygonTest(self.hull, pixel, False) >= 0:
                return True
            else:
                return False
        else:
            if cv2.pointPolygonTest(self.vertices, pixel, False) >= 0:
                return True
            else:
                return False

    def compute_sidewalk(self):
        self.compute_sidewalk_outline()
        self.fill_sidewalk()


class PathPlanner:
    """
    Class for Voronoi based path planning with A* graph search
    """
    def __init__(self, image_dimensions=(480, 640, 3)):
        self.polygon_subscriber = None
        self.image_holder = np.zeros(image_dimensions, dtype="int8")
        self.sidewalk = SidewalkPolygon(image_dimensions)
        self.plan = Plan()
        self.voronoi_diagram = None
        self.voronoi_graph = None

    def update_sidewalk(self, polygon):
        """
        Updates the sidewalk with a new set of vertices defining the polygon

        :param polygon: Vertices of the sidewalk polygon
        :return:
        """
        self.sidewalk.set_vertices(polygon)
        self.sidewalk.compute_sidewalk()

    def compute_voronoi_diagram(self, augment=False, use_hull=True, plot=True):
        """
        Computes the Voronoi diagram of the current sidewalk polygon

        :param augment:  Whether or not to augment the polygon with extra points (hardcoded)
        :param use_hull: Whether to use the convex hull of the sidewalk or original vertices
        :param plot:     Whether to plot the result
        :return:
        """
        # Whether to use the convex hull or the vertices of the sidewalk
        if use_hull:
            seeds = self.sidewalk.hull
        else:
            seeds = self.sidewalk.vertices

        # Augment polygon with image corners, this can help construct central path
        if augment:
            corners = np.asarray([[0, 0], [0, 480], [640, 480], [640, 0], [320, -100]])
            self.voronoi_diagram = Voronoi(np.vstack((seeds, corners)))
        else:
            self.voronoi_diagram = Voronoi(seeds)

        if plot:
            fig = voronoi_plot_2d(self.voronoi_diagram)
            plt.ylim([-5, 5])
            plt.xlim([-5, 5])
            plt.show()

    def generate_voronoi_graph(self):
        """
        Generates the graph structure from a Voronoi diagram and fills in id, position, and children attributes

        :return:
        """
        # TODO: Consider moving to Graph constructor
        # TODO: Generate only for vertices in sidewalk
        self.voronoi_graph = Graph()

        # Add node ids and positions
        for node_index in range(self.voronoi_diagram.vertices.shape[0]):
            self.voronoi_graph.add_node(Node(node_index))
            self.voronoi_graph.nodes[node_index].position = self.voronoi_diagram.vertices[node_index]

        # Create connections based on Voronoi ridges
        for ridge in self.voronoi_diagram.ridge_vertices:
            if ridge[0] != -1:
                self.voronoi_graph.add_edge(ridge[0], ridge[1])

    def get_vertices_in_sidewalk(self):
        """
        Finds all vertices in Voronoi diagram that are in the sidewalk

        :return: n x 1 numpy array containing indices of n points in sidewalk polygon
        """
        in_sidewalk_vertices = list()
        for node in self.voronoi_graph.nodes:
            if self.sidewalk.is_in_sidewalk(self.voronoi_diagram.vertices[node], False):
                in_sidewalk_vertices.append(node)

        return np.asarray(in_sidewalk_vertices)

    def find_start_end(self):
        """
        Finds start and goal vertices in Voronoi diagram

        :rtype: int,int
        :return: start vertex index, goal vertex index
        """
        pts_in_sidewalk = self.get_vertices_in_sidewalk()
        # Assume robot position is at bottom-center of image
        # TODO: Interpolate bottom 2 points in vertex as robot position?
        # robot_position = np.asarray([320, 480])
        robot_position = np.asarray([0, 0])

        # Handles special cases: 0 or 1 Voronoi vertices only
        if len(pts_in_sidewalk) == 0:
            start, end = None, None
            return start, end

        elif len(pts_in_sidewalk) == 1:
            start, end = pts_in_sidewalk[0], pts_in_sidewalk[0]
            return start, end

        else:
            # Sorts Voronoi vertices in sidewalk by distance to assumed robot position
            sorted_points = sorted(pts_in_sidewalk,
                                   key=lambda pt_i: np.linalg.norm(robot_position-self.voronoi_diagram.vertices[pt_i]))

        start, end = sorted_points[0], sorted_points[len(sorted_points)-1]

        return start, end

    def plan_path(self):
        """
        Path planning activation function. Starts by calling the Voronoi generation, then determines the start and goal
        positions and proceeds with an A* graph search to find a path between the nodes, before finally constructing the
        path by adding orientations to the positions.

        :rtype: bool
        :return: Returns true if a path was planned, otherwise False
        """
        self.plan.clear_plan()
        # Prepare Voronoi data structures
        self.compute_voronoi_diagram()
        self.generate_voronoi_graph()

        # Determine start and goal node in graph
        start_node_index, goal_node_index = self.find_start_end()
        path = list()

        # Handle special cases (0 or 1 vertices)
        if start_node_index is None:
            path = None
        elif start_node_index == goal_node_index:
            path = [self.voronoi_graph.nodes[start_node_index]]
        else:
            g_func = lambda node: np.linalg.norm(node.position - node.backpointer.position) + node.backpointer.g
            h_func = lambda node: np.linalg.norm(node.position - self.voronoi_graph.nodes[goal_node_index].position)
            path = a_star(self.voronoi_graph, g_func, h_func, start_node_index, goal_node_index)

        # Builds the path by extracting positions
        if isinstance(path, list):
            path = [node.position for node in path]
        else:
            self.plan.clear_plan()
            print("Error: No feasible path")
            return False

        # Orientation at each point in the path
        for i in range(0, len(path)-1):
            theta = vector_angle(path[i], path[i+1])
            r = [0, 0, theta]
            self.plan.add_pose(Pose(path[i], r))
            # At the final position in plan, use same orientation as for previous point
            # TODO: Use orientation of outgoing ridge
            if i == len(path)-2:
                self.plan.add_pose(Pose(path[i+1], r))

        if len(path) == 1:
            theta = 0
            r = [0, 0, theta]
            self.plan.add_pose(Pose(path[0], r))

        # Successfully planned a path
        return True


def vector_angle(pt1, pt2):
    """
    Given a start point and end point of a 2D vector, calculates the rotation around the z-axis

    :rtype:     float
    :param pt1: Vector start point (x,y)
    :param pt2: Vector end point (x,y)
    :return:    Vector rotation around z-axis (angle from x-axis)
    """
    v = pt2 - pt1
    return np.arctan2(v[1], v[0])


def a_star(graph, g, h, start_node_key, goal_node_key):
    """
    General A* search algorithm with customizable cost f = g(node) + h(node)

    :param graph:           graph to search, expected to be of type Graph()
    :param g:               path length function, g(n1) = path_length(n1)
    :param h:               heuristic function, h(n1) = distance_to_goal(n1)
    :param start_node_key:  key for selecting start node in graph
    :param goal_node_key:   key for selecting goal node in graph
    :return:                nodes making up the path
    """
    open_set = list()
    closed_set = list()
    path = list()
    goal_node = graph.nodes[goal_node_key]
    start_node = graph.nodes[start_node_key]
    open_set.append(start_node)

    while len(open_set) > 0:
        # Sorts the nodes in the open set by cost, see Node.__lt__()
        open_set.sort()

        # Select lowest cost node and add to closed set
        current_node = open_set.pop(0)
        closed_set.append(current_node)

        # Check exit condition
        if current_node == goal_node:
            # Construct and return path
            while current_node != start_node:
                path.append(current_node)
                current_node = current_node.backpointer
            path.append(start_node)
            return path[::-1]

        # Iterate over adjacency list
        for adjacent_node_index in current_node.children:
            adjacent_node = graph.nodes[adjacent_node_index]

            # Skip if adjacent node is in closed set
            if adjacent_node in closed_set:
                continue

            # Calculate cost of each adjacent node
            temp_h = h(adjacent_node)
            temp_g = g(adjacent_node)
            temp_f = temp_h + temp_g

            # Check if adjacent node is not in open set
            if adjacent_node not in open_set:
                # Set the cost for future reference
                adjacent_node.h = temp_h
                adjacent_node.g = temp_g
                adjacent_node.f = temp_f

                open_set.append(adjacent_node)
                adjacent_node.backpointer = current_node

            # If node is in open set, check if it has a lower cost than the current path
            elif open_set[open_set.index(adjacent_node)].f >= temp_f:
                # Update the cost for future reference
                adjacent_node.h = temp_h
                adjacent_node.g = temp_g
                adjacent_node.f = temp_f

                # Update backpointer for path construction if current path has lower cost
                adjacent_node.backpointer = current_node

    print("A*: No path found")
    return None


def random_sampling(data, num_samples):
    """
    Returns a random subset of a given dataset

    :param data: numpy.array with data to sample from, can be raw data or indices of another array
    :param num_samples: number of samples
    :return:
    """
    random_sample = np.random.randint(0, data.shape[1], size=num_samples)
    return data[random_sample, :]


def path_coverage(path, coverage_polygon, yolact_sidewalk, ground_truth):
    """
    Calculates approximate coverage of sidewalk

    :param path:             Planned path
    :param coverage_polygon: Approximate defect detector coverage in single frame
    :param yolact_sidewalk:  Sidewalk polygon as hypothesized by YOLACT
    :param ground_truth:     Ground truth sidewalk to calculate coverage of
    :rtype:                  float
    :return coverage:        The coverage ratio [0.0 : 1.0]
    """
    lookahead = 0.5
    cover = Polygon(np.asarray([[0.52, 1.08/2], [0.52, -1.08/2], [1.02, -1.84/2], [1.02, 1.84/2]]))
    sidewalk_poly = Polygon(ground_truth)
    yolact_poly = Polygon(yolact_sidewalk)

    path_points = list()
    for pose in path.poses:
        path_points.append(np.asarray([pose.x, pose.y]))
    path_points.insert(0, np.asarray([0, 0]))
    robot_start_pose = Pose([0, 0],
                            [0, 0, vector_angle(path_points[0], path_points[1])]
                            )

    path_copy = copy.deepcopy(path)
    path_copy.poses.insert(0, robot_start_pose)
    coverage_frames = list()

    for i in range(0, len(path_copy.poses)-1):
        # Interpolate the current path section
        x_range = np.linspace(path_copy.poses[i].x, path_copy.poses[i+1].x, 100, endpoint=True)
        y_range = np.linspace(path_copy.poses[i].y, path_copy.poses[i+1].y, 100, endpoint=True)
        # Compute cover position at each point
        for j in range(0, len(x_range)):
            # See shapely documentation for explanation of formatting
            transform_matrix = [np.cos(path_copy.poses[i].theta),
                                -np.sin(path_copy.poses[i].theta),
                                np.sin(path_copy.poses[i].theta),
                                np.cos(path_copy.poses[i].theta),
                                x_range[j],
                                y_range[j]]
            coverage_frames.append(affine_transform(cover, transform_matrix))


    total_cover = unary_union(coverage_frames)
    covered_sidewalk = sidewalk_poly.intersection(total_cover)

    # Plotting
    sp = PolygonPatch(sidewalk_poly, facecolor=(1, 0, 0, 0.3))
    sp2 = PolygonPatch(sidewalk_poly, facecolor=(1, 0, 0, 0.3))
    sp3 = PolygonPatch(sidewalk_poly, facecolor=(1, 0, 0, 0.3))
    yp = PolygonPatch(yolact_poly, facecolor=(0, 0.5, 0, 0.3))
    cp = PolygonPatch(covered_sidewalk, facecolor=(0,0,1,0.3))
    tcp = PolygonPatch(total_cover, facecolor=(1,0,0,1))

    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    ax1.plot([x for x,y in path_points], [y for x,y in path_points], 'blue', linestyle='--')
    ax1.add_patch(sp)
    ax1.add_patch(yp)
    ax1.autoscale_view()
    ax1.set_aspect('equal')

    ax2.plot([x for x, y in path_points], [y for x, y in path_points], 'blue', linestyle='--')
    ax2.add_patch(sp2)
    ax2.add_patch(tcp)
    ax2.autoscale_view()
    ax2.set_aspect('equal')

    ax3.plot([x for x, y in path_points], [y for x, y in path_points], 'blue', linestyle='--')
    ax3.add_patch(sp3)
    ax3.add_patch(cp)
    ax3.autoscale_view()
    ax3.set_aspect('equal')

    plt.show()

    coverage = covered_sidewalk.area / sidewalk_poly.area
    return coverage


def main():
    sidewalk_planner = PathPlanner()

    plan_publisher = rospy.Publisher('move_base/GlobalPlanner/global_plan', Path, queue_size=1)
    hotfix_publisher = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=1)
    #rospy.init_node('sidewalk_planner')

    test_i = 0

    while not rospy.is_shutdown():
        ground_truth = np.asarray(sidewalk_xyz[test_i])[:,0:2]
        yolact_extraction = np.asarray(yolact_xyz[test_i])[:,0:2]
        test_i += 1

        sidewalk_planner.update_sidewalk(yolact_extraction)
        planned = sidewalk_planner.plan_path()
        if planned:
            coverage = path_coverage(sidewalk_planner.plan, None, yolact_extraction, ground_truth)
            print(coverage)
            # msg = sidewalk_planner.plan.to_ros_msg()
            # plan_publisher.publish(msg)
            """
            # The shittiest hotfix of them all
            for i in range(len(sidewalk_planner.plan.poses)):
                hotfix_publisher.publish(sidewalk_planner.plan.poses[i].to_ros_msg())
                f = input("Press a key when the current goal is reached to publish next via-point")
            """

        # Draw lines (image space)
        """
        for i in range(len(sidewalk_planner.plan.poses) - 1):
            pt1 = (int(np.round(sidewalk_planner.plan.poses[i].x, 0)),
                   int(np.round(sidewalk_planner.plan.poses[i].y, 0)))
            pt2 = (int(np.round(sidewalk_planner.plan.poses[i+1].x, 0)),
                   int(np.round(sidewalk_planner.plan.poses[i+1].y, 0)))
            if i == 0:
                extra = (320, 480)
                cv2.line(sidewalk_planner.sidewalk.sidewalk, extra, pt1, (0, 0, 255), thickness=2)
            cv2.line(sidewalk_planner.sidewalk.sidewalk, pt1, pt2, (0, 0, 255), thickness=2)

        cv2.imshow("Tee hee", sidewalk_planner.sidewalk.sidewalk)
        cv2.waitKey(1)
        """
        rospy.loginfo("Iterated once!")
        pause = input()

    print(1)


if __name__ == '__main__':
    main()
