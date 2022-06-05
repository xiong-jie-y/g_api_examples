import os
import time

import click
import cv2 as cv

import colors
import emotion_ferplus
import facial_landmarks_35_adas_0002
import gaze_estimation_adas_0002
import head_pose_estimation_adas_0001
import open_closed_eye_0001
import sixdrepnet
import ultraface
import yunet

import faulthandler; faulthandler.enable()

# This node is necessary to let framework enable to determine type.
# This will be solved in the future.
@cv.gapi.op('custom.Dummy',
            in_types=[cv.GArray.Rect, cv.GArray.Rect, cv.GArray.Point, cv.GArray.Point],
            out_types=[cv.GArray.GMat])
class Dummy:
    @staticmethod
    def outMeta(test, test2, test3, test4):
        return cv.GMatDesc(-1,-1,[-1,-1])

@cv.gapi.kernel(Dummy)
class DummyImpl:
    @staticmethod
    def run(test, test2, test3, test4):
        return []


@click.command()
@click.option("--input", required=True)
@click.option("--detector", required=True)
@click.option("--head-pose-estimator", required=True)
@click.option("--landmark-estimator", required=True)
def main(input, detector, head_pose_estimator, landmark_estimator):
    g_in = cv.GMat()

    sz = cv.gapi.streaming.size(g_in)

    # Detect faces
    if detector == "ultraface":
        bboxes = ultraface.process(g_in)
    elif detector == "yunet":
        bboxes = yunet.process(g_in)
    elif detector == "yolo_v5_face":

        raise RuntimeError("No Such Detector")

    # Detect poses
    if head_pose_estimator == "sixdrepnet":
        angles_r, angles_p, angles_y = sixdrepnet.process(g_in, bboxes) 
        heads_pos = sixdrepnet.GProcessPoses.on(angles_y, angles_p, angles_r)
    elif head_pose_estimator == "head_pose_estimation_adas_0001":
        angles_y, angles_p, angles_r = head_pose_estimation_adas_0001.process(g_in, bboxes)
        heads_pos = head_pose_estimation_adas_0001.GProcessPoses.on(angles_y, angles_p, angles_r)
    else:
        raise RuntimeError("No Such Head Pose Detector")

    if landmark_estimator == "facial_landmarks_35_adas_0002":
        landmark = facial_landmarks_35_adas_0002.process(g_in, bboxes)
        left_eyes, right_eyes, mids, lmarks = \
            facial_landmarks_35_adas_0002.GParseEyes.on(landmark, bboxes, sz)
    else:
        raise RuntimeError("No Such facial landmark estimator")

    l_eye_st, r_eye_st = open_closed_eye_0001.process(g_in, left_eyes, right_eyes)

    emotion_id = emotion_ferplus.process(g_in, bboxes)

    gaze_vectors = gaze_estimation_adas_0002.process(g_in, left_eyes, right_eyes, heads_pos)

    # Dummy
    test = Dummy.on(left_eyes, right_eyes, mids, lmarks)

    out = cv.gapi.copy(g_in)

    computation = cv.GComputation(cv.GIn(g_in), cv.GOut(
        out, bboxes, left_eyes, right_eyes, gaze_vectors, 
        angles_y, angles_p, angles_r, mids, lmarks, 
        emotion_id, l_eye_st, r_eye_st, test))

    compiled_computation = computation.compileStreaming(args=cv.gapi.compile_args(
        cv.gapi.kernels(
            *open_closed_eye_0001.kernels(), *emotion_ferplus.kernels(), 
            *sixdrepnet.kernels(), *ultraface.kernels(),*yunet.kernels(), 
            *head_pose_estimation_adas_0001.kernels(),
            *facial_landmarks_35_adas_0002.kernels(), 
            *gaze_estimation_adas_0002.kernels(),
            DummyImpl),
        cv.gapi.networks(
            *open_closed_eye_0001.networks(), *emotion_ferplus.networks(), 
            *sixdrepnet.networks(), *ultraface.networks(), *yunet.networks(), 
            *head_pose_estimation_adas_0001.networks(), 
            *facial_landmarks_35_adas_0002.networks(), 
            *gaze_estimation_adas_0002.networks(),
            )))

    # Set input and start.
    if input.isdigit():
        source = cv.gapi.wip.make_capture_src(int(input))
    else:
        if any([os.path.splitext(input)[1] == ext for ext in [".jpg", ".png", ".bmp"]]):
            source = cv.imread(input)
        else:
            source = cv.gapi.wip.make_capture_src(input)

    compiled_computation.setSource(cv.gin(source))
    compiled_computation.start()

    frames = 0
    fps = 0
    print('Processing')
    START_TIME = time.time()

    while True:
        # time.sleep(0.3)
        start_time_cycle = time.time()
        has_frame, (oimg,
                    outr,
                    l_eyes,
                    r_eyes,
                    outg,
                    out_y,
                    out_p,
                    out_r,
                    out_mids,
                    outl, emotion_id, out_st_l,
                    out_st_r, test) = compiled_computation.pull()

        if not has_frame:
            break

        if outr is None:
            continue

        for i, out_rect in enumerate(outr):
            # Face box
            # cv.rectangle(oimg, out_rect, RED, 1)

            rx, ry, rwidth, rheight = out_rect

            # Landmarks
            facial_landmarks_35_adas_0002.draw_landmarks(
                oimg, facial_landmarks_35_adas_0002.get_nth_landmarks(outl, i), rwidth)

            # Headposes
            yaw = out_y[i]
            pitch = out_p[i]
            roll = out_r[i]
            x_center = int(rx + rwidth / 2)
            y_center = int(ry + rheight / 2)

            if head_pose_estimator == "sixdrepnet":
                # sixdrepnet.plot_pose_cube(oimg, yaw, pitch ,roll, x_center, y_center, size=rwidth)
                sixdrepnet.draw_axis(oimg, yaw, pitch, roll, x_center, y_center)
            elif head_pose_estimator == "head_pose_estimation_adas_0001":
                head_pose_estimation_adas_0001.draw_axis(oimg, yaw, pitch, roll, x_center, y_center, rwidth)

            gaze_estimation_adas_0002.visualize_angle_text(oimg, out_rect, outg[i])
            emotion_ferplus.visualize_emotion_label(oimg, out_rect, emotion_id[i])

            # Eyes boxes
            open_closed_eye_0001.visualize_eye_status(oimg, l_eyes[i], out_st_l[i])
            open_closed_eye_0001.visualize_eye_status(oimg, r_eyes[i], out_st_r[i])

            # Gaze vectors
            if out_st_l[i]:
                gaze_estimation_adas_0002.visualize_eye_direction(
                    oimg, out_mids[0 + i * 2], outg[i], rwidth)
            if out_st_r[i]:
                gaze_estimation_adas_0002.visualize_eye_direction(
                    oimg, out_mids[1 + i * 2], outg[i], rwidth)


        # Add FPS value to frame
        cv.putText(oimg, "FPS: %0i" % (fps), [int(20), int(40)],
                   cv.FONT_HERSHEY_PLAIN, 2, colors.RED, 2)

        # Show result
        cv.imshow('Gaze Estimation', oimg)
        cv.waitKey(1)

        fps = int(1. / (time.time() - start_time_cycle))
        frames += 1
    EXECUTION_TIME = time.time() - START_TIME
    print('Execution successful')
    print('Mean FPS is ', int(frames / EXECUTION_TIME))
    
if __name__ == "__main__":
    main()
