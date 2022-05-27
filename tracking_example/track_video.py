import os
import time
import numpy as np
import cv2 as cv
import click

@cv.gapi.op('custom.BoxSelector', 
    in_types=[cv.GMat], out_types=[cv.GOpaque.Rect])
class BoxSelector:
    @staticmethod
    def outMeta(image):
        return cv.empty_gopaque_desc()

@cv.gapi.kernel(BoxSelector)
class BoxSelectorImpl:
    def __init__(self):
        self.previous_bbox = None

    @staticmethod
    def setup(image):
        return BoxSelectorImpl()

    @staticmethod
    def run(image, state):
        if state.previous_bbox is None:
            bbox = cv.selectROI("Tracker", image)
            state.previous_bbox = bbox
            return bbox
        else:
            return state.previous_bbox

@cv.gapi.op('custom.Tracker',
            in_types=[cv.GMat, cv.GOpaque.Rect],
            out_types=[cv.GOpaque.Rect])
class Tracker:
    @staticmethod
    def outMeta(input_image_desc, box):
        return input_image_desc

class State:
    def __init__(self, tracker, initial_box):
        self.tracker = tracker
        self.initial_box = initial_box

@cv.gapi.kernel(Tracker)
class TrackerImpl:
    @staticmethod
    def setup(input_image_desc, box):
        tracker = cv.TrackerKCF_create()
        return State(tracker, None)

    @staticmethod
    def run(image, box, state):
        if state.initial_box is None:
            state.tracker.init(image, box)
            state.initial_box = box
            return box
        else:
            ok, bbox = state.tracker.update(image)
            return bbox

@click.command()
@click.option("--input", help="The device id or video file path or image file path.")
def main(input):
    RED = (0, 0, 255)

    # Define graph structure.
    g_in = cv.GMat()
    out = Tracker.on(g_in, BoxSelector.on(g_in))
    img_out = cv.gapi.copy(g_in)
    comp = cv.GComputation(cv.GIn(g_in), cv.GOut(img_out, out))

    ccomp = comp.compileStreaming(args=cv.gapi.compile_args(
        cv.gapi.kernels(BoxSelectorImpl, TrackerImpl)))

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
        has_frame, (debug_image, bbox) = ccomp.pull()

        time.sleep(0.033)

        if not has_frame:
            break

        # Add FPS value to frame.
        cv.putText(debug_image, "FPS: %0i" % (fps), [int(20), int(40)],
                   cv.FONT_HERSHEY_PLAIN, 2, RED, 2)

        cv.rectangle(debug_image, bbox, (0, 1, 0), thickness=2)
        cv.imshow('Tracker', debug_image)

        key = cv.waitKey(1)
        if key == 27:
            break

        fps = int(1. / (time.time() - start_time_cycle))

    print("Waiting to finish.")
    ccomp.stop()
    print("Finished")

if __name__ == '__main__':
    main()