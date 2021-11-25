import math
from math import pi, cos, sin

import numpy as np
from geometry_msgs.msg import Quaternion

def vector3_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])

def quat_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])

def point_to_numpy(msg):
    return np.array([msg.x, msg.y, msg.z])


def euler_from_quaternion(orientation):
    x, y, z, w = orientation

    roll = math.atan2( 2*w*x + 2*y*z, 1 - 2*x*x - 2*y*y )
    pitch = math.asin( max( -1, min( 1 , 2*w*y - 2*z*x ) ) )
    yaw = math.atan2( 2*w*z + 2*x*y, 1 - 2*y*y - 2*z*z )

    # r = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    # p = math.asin(2 * (w * y - z * x))
    # y = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))

    return roll, pitch, yaw

def euler_to_quaternion(euler_angles):
    roll, pitch, yaw  = euler_angles

    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)

    q = Quaternion()
    q.w = cr * cp * cy + sr * sp * sy
    q.x = sr * cp * cy - cr * sp * sy
    q.y = cr * sp * cy + sr * cp * sy
    q.z = cr * cp * sy - sr * sp * cy

    return q