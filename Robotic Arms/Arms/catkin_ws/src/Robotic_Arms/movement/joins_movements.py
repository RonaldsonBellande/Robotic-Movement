import rospy
from mpmath import *
import tf
from sympy import *


def Rotation_x_axis(theta):
    rotation_x_axis = Matrix([[ 1,                  0,            0],
                              [ 0,         cos(theta),  -sin(theta)],
                              [ 0,         sin(theta),  cos(theta)]])
    return rotation_x_axis

def Rotation_y_axis(theta):
    rotation_y_axis = Matrix([[ cos(theta),        0,  sin(theta)],
                              [          0,        1,           0],
                              [-sin(theta),        0, cos(theta)]])
    return rotation_y_axis

def Rotation_z_axis(theta):
    rotation_z_axis = Matrix([[ cos(theta),  -sin(theta),       0],
                              [ sin(theta),   cos(theta),       0],
                              [          0,            0,      1]])
    return rotation_z_axis
        

def Transformation(theta, x_position, y_position, z_position):
    transformation = Matrix([[            cos(z_position),           -sin(z_position),           0,             x_position],
                             [ sin(z_position)*cos(theta), cos(z_position)*cos(theta), -sin(theta), -sin(theta)*y_position],
                             [ sin(z_position)*sin(theta), cos(z_position)*sin(theta),  cos(theta),  cos(theta)*y_position],
                             [                          0,                          0,           0,                     1]])
