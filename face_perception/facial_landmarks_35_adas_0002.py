import cv2 as cv
import numpy as np

NUM_LANDMARK_POINTS = 35
YELLOW = (0, 255, 255)

def get_nth_landmarks(landmarks, i):
    return landmarks[NUM_LANDMARK_POINTS * i:NUM_LANDMARK_POINTS * (i+1)]

def draw_landmarks(oimg, landmarks, rwidth):
    lm_radius = int(0.01 * rwidth + 1)
    for j in range(NUM_LANDMARK_POINTS):
        cv.circle(oimg, landmarks[j], lm_radius, YELLOW, -1)

@cv.gapi.op('custom.GParseEyes',
            in_types=[cv.GArray.GMat, cv.GArray.Rect, cv.GOpaque.Size],
            out_types=[cv.GArray.Rect, cv.GArray.Rect, cv.GArray.Point, cv.GArray.Point])
class GParseEyes:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1, arr_desc2):
        return cv.empty_array_desc(), cv.empty_array_desc(), cv.empty_array_desc(), cv.empty_array_desc()

# ------------------------Support functions for custom kernels------------------------
def intersection(surface, rect):
    """ Remove zone of out of bound from ROI

    Params:
    surface: image bounds is rect representation (top left coordinates and width and height)
    rect: region of interest is also has rect representation

    Return:
    Modified ROI with correct bounds
    """
    l_x = max(surface[0], rect[0])
    l_y = max(surface[1], rect[1])
    width = min(surface[0] + surface[2], rect[0] + rect[2]) - l_x
    height = min(surface[1] + surface[3], rect[1] + rect[3]) - l_y
    if width < 0 or height < 0:
        return (0, 0, 0, 0)
    return (l_x, l_y, width, height)


def process_landmarks(r_x, r_y, r_w, r_h, landmarks):
    """ Create points from result of inference of facial-landmarks network and size of input image

    Params:
    r_x: x coordinate of top left corner of input image
    r_y: y coordinate of top left corner of input image
    r_w: width of input image
    r_h: height of input image
    landmarks: result of inference of facial-landmarks network

    Return:
    Array of landmarks points for one face
    """
    lmrks = landmarks[0]
    raw_x = lmrks[::2] * r_w + r_x
    raw_y = lmrks[1::2] * r_h + r_y
    return np.array([[int(x), int(y)] for x, y in zip(raw_x, raw_y)])


def eye_box(p_1, p_2, scale=1.8):
    """ Get bounding box of eye

    Params:
    p_1: point of left edge of eye
    p_2: point of right edge of eye
    scale: change size of box with this value

    Return:
    Bounding box of eye and its midpoint
    """

    size = np.linalg.norm(p_1 - p_2)
    midpoint = (p_1 + p_2) / 2
    width = scale * size
    height = width
    p_x = midpoint[0] - (width / 2)
    p_y = midpoint[1] - (height / 2)
    return (int(p_x), int(p_y), int(width), int(height)), list(map(int, midpoint))


@cv.gapi.kernel(GParseEyes)
class GParseEyesImpl:
    """ Custom kernel. Get information about eyes
    """
    @staticmethod
    def run(in_landm_per_face, in_face_rcs, frame_size):
        """ Сustom kernel executable code

        Params:
        in_landm_per_face: landmarks from inference of facial-landmarks network for each face
        in_face_rcs: bounding boxes for each face
        frame_size: size of input image

        Return:
        Arrays of ROI for left and right eyes, array of midpoints and
        array of landmarks points
        """
        left_eyes = []
        right_eyes = []
        midpoints = []
        lmarks = []
        surface = (0, 0, *frame_size)
        for landm_face, rect in zip(in_landm_per_face, in_face_rcs):
            points = process_landmarks(*rect, landm_face)
            lmarks.extend(points)

            rect, midpoint_l = eye_box(points[0], points[1])
            left_eyes.append(intersection(surface, rect))

            rect, midpoint_r = eye_box(points[2], points[3])
            right_eyes.append(intersection(surface, rect))

            midpoints.append(midpoint_l)
            midpoints.append(midpoint_r)

        # lmarks = [lmark.astype(np.int32) for lmark in lmarks]
        # midpoints = [np.array(midpoint).astype(np.int32) for midpoint in midpoints]
        # import IPython; IPython.embed()
        print(left_eyes,  right_eyes)
        return left_eyes, right_eyes, midpoints, lmarks


def process(g_in, bboxes):
    landmark_inputs = cv.GInferInputs()
    landmark_inputs.setInput('data', g_in)
    landmark_outputs = cv.gapi.infer('facial-landmarks', bboxes, landmark_inputs)
    landmark = landmark_outputs.at('tf.identity')
    return landmark

def kernels():
    return [GParseEyesImpl]

def networks():
    return [cv.gapi.onnx.params(
        'facial-landmarks', "models/facial_landmarks_35_adas_0002.onnx",
        ["CUDAExecutionProvider"]).cfgNormalize([False])]