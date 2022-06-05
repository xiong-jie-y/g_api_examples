import cv2 as cv
import colors

def visualize_eye_status(oimg, rect, status):
    color_l = colors.GREEN if status else colors.RED
    cv.rectangle(oimg, rect, color_l, 1)

@cv.gapi.op('custom.open_closed_eye_0001.GGetStates',
            in_types=[cv.GArray.GMat, cv.GArray.GMat],
            out_types=[cv.GArray.Int, cv.GArray.Int])
class GGetStates:
    @staticmethod
    def outMeta(arr_desc0, arr_desc1):
        return cv.empty_array_desc(), cv.empty_array_desc()

@cv.gapi.kernel(GGetStates)
class GGetStatesImpl:
    """ Custom kernel. Get state of eye - open or closed
    """
    @staticmethod
    def run(eyesl, eyesr):
        """ Сustom kernel executable code

        Params:
        eyesl: result of inference of open-closed-eye network for left eye
        eyesr: result of inference of open-closed-eye network for right eye

        Return:
        States of left eyes and states of right eyes
        """
        out_l_st = [int(st) for eye_l in eyesl for st in (eye_l[:, 0] < eye_l[:, 1]).ravel()]
        out_r_st = [int(st) for eye_r in eyesr for st in (eye_r[:, 0] < eye_r[:, 1]).ravel()]
        return out_l_st, out_r_st


def process(g_in, left_eyes, right_eyes):
    eyes_inputs = cv.GInferInputs()
    eyes_inputs.setInput('input.1', g_in)
    eyesl_outputs = cv.gapi.infer('open-closed-eye', left_eyes, eyes_inputs)
    eyesr_outputs = cv.gapi.infer('open-closed-eye', right_eyes, eyes_inputs)
    eyesl = eyesl_outputs.at('19')
    eyesr = eyesr_outputs.at('19')
    
    return GGetStates.on(eyesl, eyesr)

def kernels():
    return [GGetStatesImpl]

def networks():
    return [cv.gapi.onnx.params(
        'open-closed-eye', "models/open_closed_eye_0001.onnx", ["CUDAExecutionProvider"])]