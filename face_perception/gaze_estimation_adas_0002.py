import cv2 as cv
import numpy as np
import colors

M_PI_180 = np.pi / 180
M_PI_2 = np.pi / 2
M_PI = np.pi

def visualize_angle_text(oimg, out_rect, outg_i):
    norm_gazes = np.linalg.norm(outg_i[0])
    v0, v1, v2 = outg_i[0][0][0]
    rx, ry, rwidth, rheight = out_rect

    scale_box = 0.004 * rwidth

    gaze_angles = [180 / M_PI * (M_PI_2 + np.arctan2(v2, v0)),
                    180 / M_PI * (M_PI_2 - np.arccos(v1 / norm_gazes))]
    cv.putText(oimg, "gaze angles: (h=%0.0f, v=%0.0f)" %
                (np.round(gaze_angles[0]), np.round(gaze_angles[1])),
                [int(rx), int(ry + rheight + 12 * rwidth / 100)],
                cv.FONT_HERSHEY_PLAIN, scale_box * 2, colors.WHITE, 1)

def visualize_eye_direction(oimg, eye_center, outg_i, rwidth):
    norm_gazes = np.linalg.norm(outg_i[0])
    gaze_vector = outg_i[0] / norm_gazes
    gaze_vector = gaze_vector.squeeze(0).squeeze(0)
    arrow_length = 0.4 * rwidth
    gaze_arrow = [arrow_length * gaze_vector[0], -arrow_length * gaze_vector[1]]
    right_arrow = [int(a+b) for a, b in zip(eye_center, gaze_arrow)]
    cv.arrowedLine(oimg, eye_center, right_arrow, colors.BLUE, 2)

def process(g_in, left_eyes, right_eyes, heads_pos):
    gaze_inputs = cv.GInferListInputs()
    gaze_inputs.setInput('left_eye_image:0', left_eyes)
    gaze_inputs.setInput('right_eye_image:0', right_eyes)
    gaze_inputs.setInput('head_pose_angles:0', heads_pos)
    gaze_outputs = cv.gapi.infer2('gaze-estimation', g_in, gaze_inputs)
    gaze_vectors = gaze_outputs.at('Identity:0')
    return gaze_vectors

def kernels():
    return []

def networks():
    return [cv.gapi.onnx.params(
        'gaze-estimation', "models/gaze_estimation_adas_0002.onnx",
        ["CUDAExecutionProvider"]).cfgNormalize([False])]