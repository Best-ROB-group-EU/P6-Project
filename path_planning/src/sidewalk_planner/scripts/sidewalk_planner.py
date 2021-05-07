import cv2

import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d


class GraphNode:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.edges = []
        self.h = None


class GraphEdge:
    def __init__(self, cost, node1, node2):
        self.cost = cost
        self.node1 = node1
        self.node2 = node2


class VoronoiGraph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.graph = None


class SidewalkPolygon:
    def __init__(self, grid_dimensions=(480,640)):
        self.vertices = []
        self.hull = np.asarray([0,0])
        self.sidewalk = np.zeros(grid_dimensions)
        self.sidewalk_indices = None
        self.voronoi_decomposition = None

    def set_vertices(self, v):
        # v: A list of 2x1 numpy arrays with coords [u,v]
        self.vertices = v

    def compute_sidewalk_outline(self):
        self.hull = cv2.convexHull(self.vertices)

    def fill_hull(self):
        self.sidewalk = np.zeros(self.sidewalk.shape)
        cv2.drawContours(self.sidewalk, [self.hull], -1, 255, thickness=cv2.FILLED)

    def compute_sidewalk_indices(self):
        self.sidewalk_indices = np.vstack(np.nonzero(self.sidewalk))

    def voronoi_graph(self):
        self.voronoi_decomposition = Voronoi(self.vertices)
        fig = voronoi_plot_2d(self.voronoi_decomposition)
        plt.show()

    def compute_sidewalk(self):
        self.compute_sidewalk_outline()
        self.fill_hull()
        self.compute_sidewalk_indices()


class PathPlanner:
    def __init__(self, image_dimensions=(480,640)):
        self.polygon_subscriber = None
        self.image_holder = np.zeros(image_dimensions, dtype="int8")
        self.sidewalk = SidewalkPolygon(image_dimensions)
        self.plan = None

    def update_sidewalk(self, polygon):
        self.sidewalk.set_vertices(polygon)
        self.sidewalk.compute_sidewalk()


def sample_line(line, num_samples):
    raise NotImplemented


def fit_line_to_points(points, num_samples=10, linetype="linear"):
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
    pt1 = np.asarray([30, 40])
    pt2 = np.asarray([221, 32])
    pt3 = np.asarray([400, 450])
    pt4 = np.asarray([30, 280])
    pt5 = np.asarray([93, 344])
    test_poly = np.asarray([pt1, pt2, pt3, pt4, pt5])
    sidewalk_planner.update_sidewalk(test_poly)
    sidewalk_planner.sidewalk.compute_sidewalk_outline()
    sidewalk_planner.sidewalk.fill_hull()
    sidewalk_planner.sidewalk.compute_sidewalk_indices()
    sidewalk_planner.sidewalk.voronoi_graph()

    cv2.imshow("Teehee", sidewalk_planner.sidewalk.sidewalk)
    cv2.waitKey()

if __name__ == '__main__':
    main()
