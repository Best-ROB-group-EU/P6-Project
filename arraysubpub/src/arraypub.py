#!/usr/bin/env python2.7

import rospy
from std_msgs.msg import Int32MultiArray
import numpy as np
from arraysubpub.msg import PointXY
from arraysubpub.msg import Tile
from arraysubpub.msg import Combined_Tiles



def tiles_publisher(tiles):
    combined_tiles = Combined_Tiles()
    for i in range(len(tiles)):
        c_tile = tiles[i]
        tile_msg = Tile()
        tile_msg.tile_id = c_tile[0]
        for i in range(len(c_tile[1])):
            point = PointXY()
            point.x = c_tile[1][i][0]
            point.y = c_tile[1][i][1]
            tile_msg.corners.append(point)
        combined_tiles.tiles.append(tile_msg)
    t_pub.publish(combined_tiles)

while __name__ == "__main__":
    t_pub = rospy.Publisher('tiles', Combined_Tiles, queue_size=10)
    rospy.init_node('arraypub', anonymous=True)

    tile = [[1, [[120, 180], [180, 120], [220, 280], [280, 220]]], [2, [[220, 280], [280, 220], [320, 380], [380, 320]]]]

    while 1:
        tiles_publisher(tile)
