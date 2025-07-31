import numpy as np
import jax.numpy as jnp

def quat_to_rotmat(quat):
    """
    Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix using JAX.
    Assumes the quaternion is normalized.
    """
    w, x, y, z = quat

    # Precompute repeated terms for efficiency
    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z

    wx = w * x
    wy = w * y
    wz = w * z

    xy = x * y
    xz = x * z

    yz = y * z

    rot_mat = np.array([
        [ww + xx - yy - zz, 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     ww - xx + yy - zz, 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     ww - xx - yy + zz]
    ]).flatten()

    return rot_mat

def rotmat_to_quat(mat):
    """
    Convert a 3x3 rotation matrix to a quaternion (w, x, y, z) using JAX.
    Assumes the matrix is a valid rotation matrix.
    """
    m = mat.reshape((3, 3))
    tr = m[0, 0] + m[1, 1] + m[2, 2]

    def case_tr_pos(_):
        S = jnp.sqrt(tr + 1.0) * 2  # S=4*w
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
        return jnp.array([w, x, y, z])

    def case_m00_max(_):
        S = jnp.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2  # S=4*x
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
        return jnp.array([w, x, y, z])

    def case_m11_max(_):
        S = jnp.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2  # S=4*y
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
        return jnp.array([w, x, y, z])

    def case_m22_max(_):
        S = jnp.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2  # S=4*z
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
        return jnp.array([w, x, y, z])

    quat = jnp.where(tr > 0, case_tr_pos(0),
            jnp.where((m[0, 0] > m[1, 1]) & (m[0, 0] > m[2, 2]), case_m00_max(0),
            jnp.where(m[1, 1] > m[2, 2], case_m11_max(0), case_m22_max(0))
        ))

    return quat

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

def angle_between_lines_np(p1, p2, p3, p4): 
    """ 
    Calculates the angle between two lines using NumPy. 

    Args: 
    p1, p2: Endpoints of the first line ((x1, y1), (x2, y2)). 
    p3, p4: Endpoints of the second line ((x3, y3), (x4, y4)). 

    Returns: 
    The angle in degrees between the two lines. 
    """ 
    # Create vectors from the points 
    v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]]) 
    v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]]) 

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    epsilon = 0.1


    # if norm_v1 < epsilon or norm_v2 < epsilon:
    #     # print("EXCEPTION 1", flush=True)
    #     return 0
    
    u1 = v1 / norm_v1
    u2 = v2 / norm_v2

    # 2. Handle the edge cases
    dot_product = np.dot(u1, u2)

    # if np.abs(dot_product-1)<0.001:
    #     # print("EXCEPTION 2", flush=True)
    #     return 0

    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])

    # print(f"ANGLES: {angle1, angle2} | Vs: {v1, v2} | Norm: {norm_v1, norm_v2} | d: {dot_product}", flush=True)
    

    angle_rad = angle2 - angle1

    return np.degrees(angle_rad) 



def turn_quat(eef_pos_1_init, eef_pos_2_init, eef_pos_1, eef_pos_2, tray_rot_init):
    # xy plane z-axis rotation
    p1, p2 = (eef_pos_1_init[0], eef_pos_1_init[1]), (eef_pos_2_init[0], eef_pos_2_init[1])
    p3, p4 = (eef_pos_1[0], eef_pos_1[1]), (eef_pos_2[0], eef_pos_2[1])
    z_rot = angle_between_lines_np(p1, p2, p3, p4)
    tray_rot = quaternion_multiply(tray_rot_init, rotation_quaternion(z_rot, [0, 0, 1]))

    # # xz plane y-axis rotation
    # p1, p2 = (eef_pos_1_init[0], eef_pos_1_init[2]), (eef_pos_2_init[0], eef_pos_2_init[2])
    # p3, p4 = (eef_pos_1[0], eef_pos_1[2]), (eef_pos_2[0], eef_pos_2[2])
    # y_rot = angle_between_lines_np(p1, p2, p3, p4)
    # # tray_rot = quaternion_multiply(tray_rot, rotation_quaternion(y_rot, [0, 1, 0]))

    # # yz plane x-axis rotation
    # p1, p2 = (eef_pos_1_init[1], eef_pos_1_init[2]), (eef_pos_2_init[1], eef_pos_2_init[2])
    # p3, p4 = (eef_pos_1[1], eef_pos_1[2]), (eef_pos_2[1], eef_pos_2[2])
    # x_rot = angle_between_lines_np(p1, p2, p3, p4)
    # # tray_rot = quaternion_multiply(tray_rot, rotation_quaternion(x_rot, [1, 0, 0]))

    # np.set_printoptions(precision=2, suppress=True)
    # print(f"XY-coords:  (({p1[0]:.2f}, {p1[1]:.2f}), ({p2[0]:.2f}, {p2[1]:.2f})), (({p3[0]:.2f}, {p3[1]:.2f}), ({p4[0]:.2f}, {p4[1]:.2f})) | XYZ-Axis rotation: {float(z_rot)}", flush=True)

    return tray_rot


def main():
    # # quat = np.array([0.70711, 0, -0.70711, 0])
    # quat0 = np.array([0, -0.70711, -0.70711, 0])#rotation_quaternion(90, [0, 0, 1])
    # quat1 = np.array([0, 0.70711, 0.70711, 0])
    # # print(quat0)
    # # quat1 = rotation_quaternion(-90, [0, 0, 1])#quaternion_multiply(rotation_quaternion(90, [1, 0, 0]), rotation_quaternion(90, [0, 1, 0]))
    # # quat0 = quaternion_multiply(quat0, quat1)
    # # print(quat0)

    # print(quaternion_distance(quat0, quat1))
    # # quat2 = rotation_quaternion(90, [0, 1, 0])
    # # quat0 = quaternion_multiply(quat0, quat2)
    
    # # # quat3 = quaternion_multiply(quat0, quat1)
    # # print(quat0)

    quat0 = np.array([1, 0, 0, 0])
    quat0 = quaternion_multiply(quaternion_multiply(quat0, rotation_quaternion(-180, [0, 1, 0])), rotation_quaternion(-90, [0, 0, 1]))


    mat0 = quat_to_rotmat(quat0)

    quat1 = rotmat_to_quat(mat0)

    print(quat0)
    print(mat0.reshape((3,3)))

    print(quat1)

if __name__=="__main__":
    main()