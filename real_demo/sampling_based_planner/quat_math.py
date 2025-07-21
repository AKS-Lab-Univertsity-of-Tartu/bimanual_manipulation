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
    return np.array([round(w, 5), round(x, 5), round(y, 5), round(z, 5)])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1
    x = w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1
    y = w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1
    z = w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1
    
    return np.array([round(w, 5), round(x, 5), round(y, 5), round(z, 5)])

class QuaternionOps:

    def quat_normalize(self, q):
        return q / np.linalg.norm(q)

    def quat_conjugate(self, q):
        w, x, y, z = q
        return np.array([w, -x, -y, -z])

    def quat_mul(self, q1, q2):
        # Hamilton product (q1 * q2), both in [w, x, y, z]
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return np.array([w, x, y, z])

    def slerp(self, q1, q2, t):
        # q1, q2: [w, x, y, z], normalized quaternions
        # t: scalar interpolation parameter [0..1]

        q1 = self.quat_normalize(q1)
        q2 = self.quat_normalize(q2)

        dot = np.dot(q1, q2)

        # If dot < 0, negate q2 to take shorter path
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        DOT_THRESHOLD = 0.9995

        if dot > DOT_THRESHOLD:
            # If quaternions are very close, use linear interpolation
            result = (1.0 - t) * q1 + t * q2
            return self.quat_normalize(result)

        # Use SLERP formula
        theta_0 = np.arccos(dot)        # angle between input quaternions
        sin_theta_0 = np.sin(theta_0)

        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s1 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s2 = sin_theta / sin_theta_0

        result = s1 * q1 + s2 * q2
        return self.quat_normalize(result)


def main():
    # quat = np.array([0.70711, 0, -0.70711, 0])
    quat0 = np.array([0, 0, 0, 1])#rotation_quaternion(90, [0, 0, 1])
    print(quat0)
    quat1 = rotation_quaternion(90, [0, 0, 1])#quaternion_multiply(rotation_quaternion(90, [1, 0, 0]), rotation_quaternion(90, [0, 1, 0]))
    quat0 = quaternion_multiply(quat0, quat1)
    print(quat0)
    quat2 = rotation_quaternion(90, [0, 1, 0])
    quat0 = quaternion_multiply(quat0, quat2)
    
    # quat3 = quaternion_multiply(quat0, quat1)
    print(quat0)

if __name__=="__main__":
    main()