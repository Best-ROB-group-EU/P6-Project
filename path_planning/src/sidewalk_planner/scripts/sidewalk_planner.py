import cv2
import rospy

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from nav_msgs.msg import Path


class Pose:
    """
    Class for holding a robot pose, used for the plan
    """
    def __init__(self, position, orientation):
        self.x = position[0]
        self.y = position[1]
        self.z = 0
        # Quaternion conversion from axis-angle
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
        self.sidewalk_indices = None

    def set_vertices(self, v):
        self.vertices = v

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
        if use_hull:
            cv2.drawContours(self.sidewalk, [self.hull], -1, (255, 255, 255), thickness=cv2.FILLED)
        else:
            cv2.drawContours(self.sidewalk, [self.vertices], -1, (255, 255, 255), thickness=cv2.FILLED)

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
        if use_hull:
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
    # u = [1, 0]
    return np.arccos(v[0]/(np.linalg.norm(v)))


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


def main():
    sidewalk_planner = PathPlanner()
    plan_publisher = rospy.Publisher('plan', Path, queue_size=1)
    #sidewalk_polygon_subscriber = rospy.Subscriber('sidewalk_polygon', Sidewalk)
    rospy.init_node('sidewalk_planner')

    while not rospy.is_shutdown():
        pt1 = np.asarray([0, 470])
        pt2 = np.asarray([620, 470])
        pt3 = np.asarray([500, 100])
        pt4 = np.asarray([100, 90])
        test_poly = np.asarray([pt1, pt2, pt3, pt4])

        sidewalk_planner.update_sidewalk(test_poly)
        planned = sidewalk_planner.plan_path()
        if planned:
            plan_publisher.publish(sidewalk_planner.plan.to_ros_msg())

        # Draw lines
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
        cv2.waitKey(0)
        rospy.loginfo("Iterated once!")

    print(1)


if __name__ == '__main__':
    main()
