import cv2 as cv
import numpy as np
import colors

NETWORK_NAME = "emotion_ferplus-emotion-estimation"
EMOTION_TABLE = {'neutral':0, 'happiness':1, 'surprise':2, 'sadness':3, 'anger':4, 'disgust':5, 'fear':6, 'contempt':7}
INVERSE_EMOTION_TABLE = [emotion_name for emotion_name in EMOTION_TABLE.keys()]

def visualize_emotion_label(oimg, out_rect, emotion_id):
    rx, ry, rwidth, rheight = out_rect
    scale_box = 0.004 * rwidth

    if emotion_id is not None:
        emotion_name = INVERSE_EMOTION_TABLE[emotion_id]
        cv.putText(oimg, f"emotion: {emotion_name}",
                [int(rx), int(ry + rheight + 24 * rwidth / 100)],
                cv.FONT_HERSHEY_PLAIN, scale_box * 2, colors.WHITE, 1)

@cv.gapi.op('custom.emotion_ferplus.Postprocess',
            in_types=[cv.GArray.GMat],
            out_types=[cv.GArray.Int])
class Postprocess:
    @staticmethod
    def outMeta(scores):
        return cv.empty_gopaque_desc()

def softmax(x):
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

@cv.gapi.kernel(Postprocess)
class PostprocessImpl:
    @staticmethod
    def run(scores):
        if len(scores) == 0:
            return [0]

        prob = softmax(scores)
        prob = np.squeeze(prob)
        classes = np.argsort(prob)[::-1]
        if len(classes.shape) == 1:
            return [classes[0]]
        return [classes_elm[0] for classes_elm in classes]

def process(g_in, bbox):
    head_inputs = cv.GInferInputs()
    head_inputs.setInput('Input3', g_in)
    emotion_outputs = cv.gapi.infer(NETWORK_NAME, bbox, head_inputs)
    return Postprocess.on(emotion_outputs.at("Plus692_Output_0"))

def kernels():
    return [PostprocessImpl]

def networks():
    return [cv.gapi.onnx.params(
        NETWORK_NAME, 
        "models/emotion-ferplus-8.onnx", 
        ["CUDAExecutionProvider"]).cfgNormalize([False])]