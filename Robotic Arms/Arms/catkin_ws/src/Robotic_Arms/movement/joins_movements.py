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
    
    
def calculate_pose(input):
    rospy.loginfo("The initial pose is %s for the task", len(input.poses))
    
    if len(input.poses) < 1:
        print("Not a valid pose")
        return -1
    else:
        
        x_position1, x_position2, x_position3, x_position4, x_position5, x_position6, x_position7 = symbols('x_position1:8')
        y_position1, y_position2, y_position3, y_position4, y_position5, y_position6, y_position7 = symbols('y_position1:8')
        z_position1, z_position2, z_position3, z_position4, z_position5, z_position6, z_position7 = symbols('z_position1:8')
        theta1, theta2, theta3, theta4, theta5, theta6, theta7 = symbols('theta1:8')
        
        states = {theta1:        0, x_position1:      0, y_position1:  0.75,  z_position1:       z_position1,
             theta2:    -pi/2, x_position2:   0.35, y_position2:     0,  z_position2:  z_position2-pi/2,
             theta3:        0, x_position3:   1.25, y_position3:     0,  z_position3:       z_position3,
             theta4:    -pi/2, x_position4: -0.054, y_position4:   1.5,  z_position4:       z_position1,
             theta5:     pi/2, x_position5:      0, y_position5:     0,  z_position5:       z_position5,
             theta6:    -pi/2, x_position6:      0, y_position6:     0,  z_position6:       z_position6,
             theta7:        0, x_position7:      0, y_position7: 0.303,  z_position7:       0}
        
        joint1 = Transformation(theta1, x_position1, y_position1, z_position1).subs(states)
        joint2 = Transformation(theta2, x_position2, y_position2, z_position2).subs(states)
        joint3 = Transformation(theta3, x_position3, y_position3, z_position3).subs(states)
        joint4 = Transformation(theta4, x_position4, y_position4, z_position4).subs(states)
        joint5 = Transformation(theta5, x_position5, y_position5, z_position5).subs(states)
        joint6 = Transformation(theta6, x_position6, y_position6, z_position6).subs(states)
        joint7 = Transformation(theta7, x_position7, y_position7, z_position7).subs(states)     # hand joints

        colab_joint = joint1 * joint2 * joint3 * joint4 * joint5 * joint6 * joint7
        
        joint_trajectory_list = []
        for x in xrange(0, len(input.poses)):
            
            joint_trajectory_position = JointTrajectoryPoint()
            
            px = input.poses[x].position.x
            py = input.poses[x].position.y
            pz = input.poses[x].position.z
            
            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion([input.poses[x].orientation.x, input.poses[x].orientation.y, input.poses[x].orientation.z, input.poses[x].orientation.w])
            
            
