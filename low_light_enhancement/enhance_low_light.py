import os
import numpy as np
import cv2 as cv
import click


def weight_path(model_path):
    assert model_path.endswith('.xml'), "Wrong topology path was provided"
    return model_path[:-3] + 'bin'

@cv.gapi.op('custom.Preprocess',
            in_types=[cv.GMat],
            out_types=[cv.GMat])
class Preprocess:
    @staticmethod
    def outMeta(input):
        return input.withDepth(cv.CV_32F)

@cv.gapi.kernel(Preprocess)
class PreprocessImpl:
    @staticmethod
    def run(inference_result):
        output_image = inference_result.astype('float32') / 255.0
        return output_image


@cv.gapi.op('custom.GExtractImage',
            in_types=[cv.GMat, cv.GMat],
            out_types=[cv.GMat])
class GExtractImage:
    @staticmethod
    def outMeta(input_image_desc, inference_result):
        return input_image_desc

@cv.gapi.kernel(GExtractImage)
class GExtractImageImpl:
    @staticmethod
    def run(original_image_size, inference_result):
        output_image = (inference_result * 255.0).astype(np.uint8)
        output_image = output_image.squeeze()
        output_image = np.transpose(output_image, (1, 2, 0))
        output_image = cv.resize(output_image, (original_image_size.shape[1], original_image_size.shape[0]))
        return output_image

@click.command()
@click.option("--input", help="The device id or video file path or image file path.")
def main(input):
    g_in = cv.GMat()

    raw_inputs = cv.GInferInputs()
    raw_inputs.setInput('input_low', Preprocess.on(g_in))
    enhanced_outputs = cv.gapi.infer('light-enhancement', raw_inputs)
    enhanced_outputs_2 = GExtractImage.on(g_in, enhanced_outputs.at('FG/conv2d_16/Relu'))
    out = cv.gapi.copy(g_in)

    comp = cv.GComputation(cv.GIn(g_in), cv.GOut(out, enhanced_outputs_2))

    light_enhancement_net = cv.gapi.ie.params(
        'light-enhancement', "saved_model_360x640/openvino/FP32/gladnet.xml",
        weight_path("saved_model_360x640/openvino/FP32/gladnet.xml"), 'CPU')

    if input.isdigit():
        # raise RuntimeError("This require next opencv version.")
        source = cv.gapi.wip.make_capture_src(int(input))
    else:
        if any([os.path.splitext(input)[1] == ext for ext in [".jpg", ".png", ".bmp"]]):
            source = cv.imread(input)
        else:
            source = cv.gapi.wip.make_capture_src(input)
    

    ccomp = comp.compileStreaming(args=cv.gapi.compile_args(
        cv.gapi.kernels(GExtractImageImpl, PreprocessImpl), cv.gapi.networks(light_enhancement_net)))
    ccomp.setSource(cv.gin(source))
    ccomp.start()

    while True:
        has_frame, (oimg, enhanced_img) = ccomp.pull()

        cv.imshow('Original', oimg)
        cv.imshow('Gaze Estimation', enhanced_img)
        key = cv.waitKey(1)
        if key == 27:
            break

if __name__ == '__main__':
    main()