import numpy as np

def quaternion_distance(q1, q2):
    dot_product = np.abs(np.dot(q1, q2))
    dot_product = np.clip(dot_product, -1.0, 1.0)
    return 2 * np.arccos(dot_product)

def rotation_quaternion(angle_deg, axis):
    axis = axis / np.linalg.norm(axis)
    angle_rad = np.deg2rad(angle_deg)
    w = np.cos(angle_rad / 2)
    x, y, z = axis * np.sin(angle_rad / 2)
    return (round(w, 5), round(x, 5), round(y, 5), round(z, 5))

def quaternion_multiply(q1, q2):
		w1, x1, y1, z1 = q1
		w2, x2, y2, z2 = q2
		
		w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
		x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
		y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
		z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
		
		return (round(w, 5), round(x, 5), round(y, 5), round(z, 5))