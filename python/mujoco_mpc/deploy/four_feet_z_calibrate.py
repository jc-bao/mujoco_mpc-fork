import numpy as np

def fit_plane_svd(points):
    """
    Given N points (N >= 3) in 3D space, find the unit normal of the
    best-fit plane via SVD and return (normal, centroid).
    """
    # 1) Compute centroid
    centroid = np.mean(points, axis=0)  # shape (3,)

    # 2) Shift points so that centroid is at the origin
    shifted = points - centroid  # shape (N, 3)

    # 3) SVD on the shifted points
    U, S, Vt = np.linalg.svd(shifted, full_matrices=False)
    # The plane normal is given by the last row of V^T
    normal = Vt[-1, :]

    # 4) Normalize it
    normal /= np.linalg.norm(normal)

    return normal, centroid

def rotation_matrix_from_vectors(vec_from, vec_to):
    """
    Returns the rotation matrix R that aligns vec_from to vec_to.
    Both vec_from and vec_to are 3D vectors.
    """
    # 1) Make sure both vectors are normalized
    v1 = vec_from / np.linalg.norm(vec_from)
    v2 = vec_to   / np.linalg.norm(vec_to)

    # 2) If they are (almost) the same, return identity
    dot = np.dot(v1, v2)
    if np.isclose(dot, 1.0):
        return np.eye(3)

    # 3) Cross product as rotation axis
    cross = np.cross(v1, v2)

    # 4) Construct the skew-symmetric cross-product matrix K
    K = np.array([
        [0,         -cross[2],  cross[1]],
        [cross[2],  0,         -cross[0]],
        [-cross[1], cross[0],   0       ]
    ])

    # 5) Rodrigues’ rotation formula: R = I + K + K^2 * (1 / (1 + dot))
    R = np.eye(3) + K + K @ K * (1.0 / (1.0 + dot))
    return R

def minimize_z_difference(points):
    """
    Given 4 or more points in 3D (shape (N,3)), this function:
      1) Finds the best-fit plane normal.
      2) Creates a rotation matrix R that will rotate the plane's normal
         to the z-axis about the origin (no centroid shifting).
      3) Returns (R, z_offset), where z_offset is the average z-value
         after rotation. If you rotate first (about the origin) and then
         subtract z_offset, your plane will be parallel to XY.
    """
    # 1) Fit the plane to get the normal (and centroid, which we will not shift by)
    normal, centroid = fit_plane_svd(points)

    # 2) Build rotation that takes 'normal' -> [0, 0, 1], around origin
    R = rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))

    # 3) Rotate all original points about the origin
    rotated_points = (R @ points.T).T

    # 4) The z_offset is the mean z of the rotated points
    z_offset = np.mean(rotated_points[:, 2])

    return R, z_offset


# EXAMPLE USAGE
if __name__ == "__main__":
    # Sample data: shape (4,3)
    pts = np.array([
        [1.0, 2.0, 5.0],
        [2.0, 3.0, 6.0],
        [3.0, 4.0, 7.0],
        [4.5, 1.0, 5.5]
    ])

    R, z_offset = minimize_z_difference(pts)

    # Apply rotation about the origin
    pts_rot = (R @ pts.T).T

    # Then subtract the z_offset so that the plane is near z=0
    pts_rot[:, 2] -= z_offset

    print("Original points:\n", pts)
    print("\nRotation matrix R:\n", R)
    print(f"\nZ offset: {z_offset:.4f}")

    print("\nRotated + offset points:\n", pts_rot)

    # Check the range of Z-values of the rotated points
    zs = pts_rot[:, 2]
    print("\nZ-values after rotation and offset:", zs)
    print("Range of Z after rotation + offset =", zs.max() - zs.min())
