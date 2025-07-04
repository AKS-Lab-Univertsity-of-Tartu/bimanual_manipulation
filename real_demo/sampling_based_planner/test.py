from quat_math import rotation_quaternion, quaternion_multiply
import numpy as np

q = quaternion_multiply(rotation_quaternion(90, np.array([1,0,0])), rotation_quaternion(45, np.array([0,0,1])))
print(q)
print(rotation_quaternion(90, np.array([1,0,0])))