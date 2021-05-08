import cv2

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d


class Node:
    """
    Used for constructing an undirected, unweighted graph with adjency list
    """
    def __init__(self, id, children=None):
        self.id = id
        if children == None:
            self.children = list()
        else: self.children = children

    def add_child(self, child_node):
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
        if isinstance(node, Node) and node.id not in self.nodes:
            self.nodes[node.id] = node
            return True
        else:
            return False

    def add_edge(self, n1, n2):
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
    def __init__(self, grid_dimensions=(480,640,3)):
        self.vertices = list()
        self.hull = np.asarray([0,0])
        self.sidewalk = np.zeros(grid_dimensions)
        self.sidewalk_indices = None

    def set_vertices(self, v):
        # v: A list of 2x1 numpy arrays with coords [u,v]
        self.vertices = v

    def compute_sidewalk_outline(self):
        self.hull = cv2.convexHull(self.vertices)
        self.hull = np.reshape(self.hull, (self.hull.shape[0], 2))

    def fill_sidewalk(self):
        cv2.drawContours(self.sidewalk, [self.vertices], -1, 255, thickness=cv2.FILLED)

    def fill_hull(self):
        self.sidewalk = np.zeros(self.sidewalk.shape)
        cv2.drawContours(self.sidewalk, [self.hull], -1, 255, thickness=cv2.FILLED)

    def compute_sidewalk_indices(self):
        self.sidewalk_indices = np.vstack(np.nonzero(self.sidewalk))

    def is_in_sidewalk(self, pt, use_hull=False):
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
        #self.fill_sidewalk()
        self.fill_hull()
        #self.compute_sidewalk_indices()


class PathPlanner:
    def __init__(self, image_dimensions=(480,640,3)):
        self.polygon_subscriber = None
        self.image_holder = np.zeros(image_dimensions, dtype="int8")
        self.sidewalk = SidewalkPolygon(image_dimensions)
        self.plan = list()
        self.voronoi_diagram = None
        self.voronoi_graph = None

    def update_sidewalk(self, polygon):
        self.sidewalk.set_vertices(polygon)
        self.sidewalk.compute_sidewalk()

    def compute_voronoi_diagram(self):
        # Augment polygon with image corners, this helps construct central path
        corners = np.asarray([[0,0], [0, 480], [640, 480], [640, 0], [320, 0]])
        self.voronoi_diagram = Voronoi(np.vstack((self.sidewalk.hull, corners)))
        fig = voronoi_plot_2d(self.voronoi_diagram)
        plt.ylim([-200, 800])
        plt.xlim([-200, 800])
        plt.show()

    def generate_voronoi_graph(self):
        # TODO: Consider moving to Graph constructor
        self.voronoi_graph = Graph()
        for id in range(self.voronoi_diagram.vertices.shape[0]):
            self.voronoi_graph.add_node(Node(id))

        for ridge in self.voronoi_diagram.ridge_vertices:
            if ridge[0] != -1:
                self.voronoi_graph.add_edge(ridge[0], ridge[1])

    def get_vertices_in_sidewalk(self):
        """
        Finds all vertices in Voronoi diagram that are in the sidewalk
        :return:
        """
        in_sidewalk_vertices = list()
        for node in self.voronoi_graph:
            if self.sidewalk.is_in_sidewalk(self.voronoi_diagram[node.id], False):
                in_sidewalk_vertices.append(node.id)

        return np.asarray(in_sidewalk_vertices)

    def find_start_end(self):
        pts_in_sidewalk = self.get_vertices_in_sidewalk()
        # For all vertices, if voronoi vertex is in sidewalk, find minimum distance to robots position
        # Assume robot position is at bottom-center of image
        robot_position = np.asarray([320,480])

        for node in self.voronoi_graph:
            if self.sidewalk.is_in_sidewalk(self.voronoi_diagram[node.id], False):

                print(node)
            start = 0
            self.plan[0] = start

        # For all vertices, if vertex is in sidewalk, find
        end = 1
        self.plan.append(end)

    def plan_path(self):
        raise NotImplemented


def a_star(graph, g, h):
    raise NotImplemented


def random_sampling(data, num_samples):
    """
    Returns a random subset of a given dataset

    Parameters
    -----------
    :param data: numpy.array with data to sample from, can be raw data or indices of another array
    :param samples: number of samples, if type(samples) = int will take the flat number, if float a ratio (0.1 = 10%)
    :return:
    """
    random_sample = np.random.randint(0, data.shape[1], size=num_samples)

    return data[random_sample,:]


def main():
    sidewalk_planner = PathPlanner()
    pt1 = np.asarray([0, 470])
    pt2 = np.asarray([620, 470])
    pt3 = np.asarray([500, 100])
    pt4 = np.asarray([100, 90])
    pt5 = np.asarray([93, 350])
    pt6 = np.asarray([320, 480])
    test_poly = np.asarray([pt1, pt2, pt3, pt4])

    sidewalk_planner.update_sidewalk(test_poly)
    sidewalk_planner.sidewalk.compute_sidewalk()
    sidewalk_planner.voronoi_diagram()
    sidewalk_planner.generate_voronoi_graph()
    sidewalk_planner.find_start_end()

    # Draw lines
    for i in range(sidewalk_planner.voronoi_diagram.vertices.shape[0] - 1):
        if sidewalk_planner.voronoi_diagram.ridge_vertices[i][0] != -1:
            j = sidewalk_planner.voronoi_diagram.ridge_vertices[i][0]
            k = sidewalk_planner.voronoi_diagram.ridge_vertices[i][1]
            pt1 = (int(np.round(sidewalk_planner.voronoi_diagram.vertices[j], 0)[0]),
                   int(np.round(sidewalk_planner.voronoi_diagram.vertices[j], 0)[1]))
            pt2 = (int(np.round(sidewalk_planner.voronoi_diagram.vertices[k], 0)[0]),
                   int(np.round(sidewalk_planner.voronoi_diagram.vertices[k], 0)[1]))
            cv2.line(sidewalk_planner.sidewalk.sidewalk, pt1, pt2, (0,0,255), thickness=2)
    
    cv2.imshow("Teehee", sidewalk_planner.sidewalk.sidewalk)
    cv2.waitKey()
    print(1)


if __name__ == '__main__':
    main()
