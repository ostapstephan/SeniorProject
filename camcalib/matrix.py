import numpy as np


def get_perpendicular_vector(v):
    """ Finds an arbitrary perpendicular vector to *v*."""
    # http://codereview.stackexchange.com/questions/43928/algorithm-to-get-an-arbitrary-perpendicular-vector
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        print("zero-vector")

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array((1, 0, 0))
    if v[1] == 0:
        return np.array((0, 1, 0))
    if v[2] == 0:
        return np.array((0, 0, 1))

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    return np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]])


def get_anthropomorphic_matrix():
    temp = np.identity(4)
    temp[2, 2] *= -1
    return temp


def get_adjusted_pixel_space_matrix(scale):
    # returns a homoegenous matrix
    temp = get_anthropomorphic_matrix()
    temp[3, 3] *= scale
    return temp


def get_image_space_matrix(image_width, image_height, focal_length, scale=1.0):
    temp = get_adjusted_pixel_space_matrix(scale)
    temp[1, 1] *= -1  # image origin is top left
    temp[0, 3] = -image_width / 2.0
    temp[1, 3] = image_height / 2.0
    temp[2, 3] = -focal_length
    return temp.T


def get_pupil_transformation_matrix(circle_normal, circle_center, circle_scale=1.0):
    """
        OpenGL matrix convention for typical GL software
        with positive Y=up and positive Z=rearward direction
        RT = right
        UP = up
        BK = back
        POS = position/translation
        US = uniform scale

        float transform[16];

        [0] [4] [8 ] [12]
        [1] [5] [9 ] [13]
        [2] [6] [10] [14]
        [3] [7] [11] [15]

        [RT.x] [UP.x] [BK.x] [POS.x]
        [RT.y] [UP.y] [BK.y] [POS.y]
        [RT.z] [UP.z] [BK.z] [POS.Z]
        [    ] [    ] [    ] [US   ]
    """
    temp = get_anthropomorphic_matrix()
    right = temp[:3, 0]
    up = temp[:3, 1]
    back = temp[:3, 2]
    translation = temp[:3, 3]
    back[:] = np.array(circle_normal)
    back[2] *= -1  # our z axis is inverted

    if np.linalg.norm(back) != 0:
        back[:] /= np.linalg.norm(back)
        right[:] = get_perpendicular_vector(back) / np.linalg.norm(
            get_perpendicular_vector(back)
        )
        up[:] = np.cross(right, back) / np.linalg.norm(np.cross(right, back))
        right[:] *= circle_scale
        back[:] *= circle_scale
        up[:] *= circle_scale
        translation[:] = np.array(circle_center)
        translation[2] *= -1
    return temp.T
