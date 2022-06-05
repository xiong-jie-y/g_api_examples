import cv2 as cv
import numpy as np

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
PINK = (255, 0, 255)
YELLOW = (0, 255, 255)

M_PI_180 = np.pi / 180
M_PI_2 = np.pi / 2
M_PI = np.pi

def draw_axis(oimg, yaw, pitch, roll, x_center, y_center, rwidth):
    sin_y = np.sin(yaw[:] * M_PI_180)
    sin_p = np.sin(pitch[:] * M_PI_180)
    sin_r = np.sin(roll[:] * M_PI_180)

    cos_y = np.cos(yaw[:] * M_PI_180)
    cos_p = np.cos(pitch[:] * M_PI_180)
    cos_r = np.cos(roll[:] * M_PI_180)

    axis_length = 0.4 * rwidth

    # center to right
    cv.line(oimg, [x_center, y_center],
            [int(x_center + axis_length * (cos_r * cos_y + sin_y * sin_p * sin_r)),
            int(y_center + axis_length * cos_p * sin_r)],
            RED, 2)

    # center to top
    cv.line(oimg, [x_center, y_center],
            [int(x_center + axis_length * (cos_r * sin_y * sin_p + cos_y * sin_r)),
            int(y_center - axis_length * cos_p * cos_r)],
            GREEN, 2)

    # center to forward
    cv.line(oimg, [x_center, y_center],
            [int(x_center + axis_length * sin_y * cos_p),
            int(y_center + axis_length * sin_p)],
            PINK, 2)

# ------------------------Custom graph operations------------------------
@cv.gapi.op('custom.head_pose_estimation_adas_0001.GProcessPoses',
            in_types=[cv.GArray.GMat, cv.GArray.GMat, cv.GArray.GMat],
            out_types=[cv.GArray.GMat])
class GProcessPoses:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1, arr_desc2):
        return cv.empty_array_desc()

# ------------------------Custom kernels------------------------
@cv.gapi.kernel(GProcessPoses)
class GProcessPosesImpl:
    """ Custom kernel. Processed poses of heads
    """
    @staticmethod
    def run(in_ys, in_ps, in_rs):
        """ Сustom kernel executable code

        Params:
        in_ys: yaw angle of head
        in_ps: pitch angle of head
        in_rs: roll angle of head

        Return:
        Arrays with heads poses
        """
        return [np.array([ys[0], ps[0], rs[0]]).T for ys, ps, rs in zip(in_ys, in_ps, in_rs)]


def process(g_in, bboxes):
    head_inputs = cv.GInferInputs()
    head_inputs.setInput('data', g_in)
    face_outputs = cv.gapi.infer('head-pose', bboxes, head_inputs)
    angles_y = face_outputs.at('tf.identity_2')
    angles_p = face_outputs.at('tf.identity')
    angles_r = face_outputs.at('tf.identity_1')
    return angles_y, angles_p, angles_r

def kernels():
    return [GProcessPosesImpl]

def networks():
    return [cv.gapi.onnx.params(
        'head-pose', "models/head_pose_estimation_adas_0001.onnx",
        ["CUDAExecutionProvider"]).cfgNormalize([False])]