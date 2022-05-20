import os
import time
import numpy as np
import cv2 as cv
import click


def weight_path(model_path):
    assert model_path.endswith('.xml'), "Wrong topology path was provided"
    return model_path[:-3] + 'bin'

@cv.gapi.op('custom.Normalize',
            in_types=[cv.GMat],
            out_types=[cv.GMat])
class Normalize:
    @staticmethod
    def outMeta(input):
        return input.withDepth(cv.CV_32F)

@cv.gapi.kernel(Normalize)
class NormalizeImpl:
    @staticmethod
    def run(inference_result):
        output_image = inference_result.astype('float32') / 255.0
        output_image = cv.resize(output_image, (640, 360))
        output_image = np.expand_dims(np.transpose(output_image, (2, 0, 1)), axis=0)
        return output_image

@cv.gapi.op('custom.Denormalize',
            in_types=[cv.GMat],
            out_types=[cv.GMat])
class Denormalize:
    @staticmethod
    def outMeta(inference_result):
        return inference_result

@cv.gapi.kernel(Denormalize)
class DenormalizeImpl:
    @staticmethod
    def run(inference_result):
        output_image = (inference_result * 255.0).astype(np.uint8)
        return output_image

@cv.gapi.op('custom.ConvertNCHWToOpenCV',
            in_types=[cv.GMat, cv.GMat],
            out_types=[cv.GMat])
class ConvertNCHWToOpenCV:
    @staticmethod
    def outMeta(input_image_desc, inference_result):
        return input_image_desc

@cv.gapi.kernel(ConvertNCHWToOpenCV)
class ConvertNCHWToOpenCVImpl:
    @staticmethod
    def run(original_image_size, inference_result):
        output_image = inference_result.squeeze()
        # output_image = np.transpose(output_image, (0, 1, ))
        output_image = cv.resize(output_image, (original_image_size.shape[1], original_image_size.shape[0]))
        return output_image

@click.command()
@click.option("--input", help="The device id or video file path or image file path.")
def main(input):
    """The command to enhance low illuminated part of the input image or video stream."""
    RED = (0, 0, 255)

    # Define graph structure.
    g_in = cv.GMat()

    raw_inputs = cv.GInferInputs()
    raw_inputs.setInput('input_low', Normalize.on(g_in))
    enhanced_outputs = cv.gapi.infer('light-enhancement', raw_inputs)
    enhanced_outputs_2 = ConvertNCHWToOpenCV.on(
            g_in, Denormalize.on(enhanced_outputs.at('tf.identity')))
    out = cv.gapi.copy(g_in)

    comp = cv.GComputation(cv.GIn(g_in), cv.GOut(out, enhanced_outputs_2))

    # Define models and kernels that is used as a implementation.
    light_enhancement_net = cv.gapi.onnx.params(
        'light-enhancement', "saved_model_360x640/model_float32.onnx",
        ["CUDAExecutionProvider"])
    ccomp = comp.compileStreaming(args=cv.gapi.compile_args(
        cv.gapi.kernels(ConvertNCHWToOpenCVImpl, NormalizeImpl, DenormalizeImpl), cv.gapi.networks(light_enhancement_net)))

    # Set input and start.
    if input.isdigit():
        # raise RuntimeError("This require next opencv version.")
        source = cv.gapi.wip.make_capture_src(int(input))
    else:
        if any([os.path.splitext(input)[1] == ext for ext in [".jpg", ".png", ".bmp"]]):
            source = cv.imread(input)
        else:
            source = cv.gapi.wip.make_capture_src(input)
    
    ccomp.setSource(cv.gin(source))
    ccomp.start()

    fps = 0
    while True:
        start_time_cycle = time.time()
        has_frame, (oimg, enhanced_img) = ccomp.pull()

        if not has_frame:
            break

        # Add FPS value to frame.
        cv.putText(enhanced_img, "FPS: %0i" % (fps), [int(20), int(40)],
                   cv.FONT_HERSHEY_PLAIN, 2, RED, 2)

        cv.imshow('Original', oimg)
        cv.imshow('Low Light Enhanced Image', enhanced_img)

        key = cv.waitKey(1)
        if key == 27:
            break

        fps = int(1. / (time.time() - start_time_cycle))

    print("Waiting to finish.")
    ccomp.stop()
    print("Finished")

if __name__ == '__main__':
    main()