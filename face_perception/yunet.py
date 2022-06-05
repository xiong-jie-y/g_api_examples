import cv2 as cv
import numpy as np

MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
STEPS = [8, 16, 32, 64]
VARIANCE = [0.1, 0.2]

class YuNetONNX(object):

    # Feature map用定義
    MIN_SIZES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    STEPS = [8, 16, 32, 64]
    VARIANCE = [0.1, 0.2]

    def __init__(
        self,
        input_shape=[160, 120],
        conf_th=0.6,
        nms_th=0.3,
        topk=5000,
        keep_topk=750,
    ):
        # 各種設定
        self.input_shape = input_shape  # [w, h]
        self.conf_th = conf_th
        self.nms_th = nms_th
        self.topk = topk
        self.keep_topk = keep_topk

        # priors生成
        self.priors = None
        self._generate_priors()

    def _generate_priors(self):
        w, h = self.input_shape

        feature_map_2th = [
            int(int((h + 1) / 2) / 2),
            int(int((w + 1) / 2) / 2)
        ]
        feature_map_3th = [
            int(feature_map_2th[0] / 2),
            int(feature_map_2th[1] / 2)
        ]
        feature_map_4th = [
            int(feature_map_3th[0] / 2),
            int(feature_map_3th[1] / 2)
        ]
        feature_map_5th = [
            int(feature_map_4th[0] / 2),
            int(feature_map_4th[1] / 2)
        ]
        feature_map_6th = [
            int(feature_map_5th[0] / 2),
            int(feature_map_5th[1] / 2)
        ]

        feature_maps = [
            feature_map_3th, feature_map_4th, feature_map_5th, feature_map_6th
        ]

        priors = []
        for k, f in enumerate(feature_maps):
            min_sizes = self.MIN_SIZES[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / w
                    s_ky = min_size / h

                    cx = (j + 0.5) * self.STEPS[k] / w
                    cy = (i + 0.5) * self.STEPS[k] / h

                    priors.append([cx, cy, s_kx, s_ky])

        self.priors = np.array(priors, dtype=np.float32)

    def postprocess(self, result):
        # 結果デコード
        dets = self._decode(result)

        # NMS
        keepIdx = cv.dnn.NMSBoxes(
            bboxes=dets[:, 0:4].tolist(),
            scores=dets[:, -1].tolist(),
            score_threshold=self.conf_th,
            nms_threshold=self.nms_th,
            top_k=self.topk,
        )

        # import IPython; IPython.embed()

        # bboxes, landmarks, scores へ成形
        scores = []
        bboxes = []
        landmarks = []
        # import IPython; IPython.embed()
        if len(keepIdx) > 0:
            dets = dets[keepIdx]
            if len(dets.shape) == 3:
                dets = np.squeeze(dets, axis=1)
            for det in dets[:self.keep_topk]:
                scores.append(det[-1])
                bboxes.append(det[0:4])
                landmarks.append(det[4:14].reshape((5, 2)))

        return bboxes, landmarks, scores

    def _decode(self, result):
        loc, conf, iou = result

        # スコア取得
        cls_scores = conf[:, 1]
        iou_scores = iou[:, 0]

        _idx = np.where(iou_scores < 0.)
        iou_scores[_idx] = 0.
        _idx = np.where(iou_scores > 1.)
        iou_scores[_idx] = 1.
        scores = np.sqrt(cls_scores * iou_scores)
        scores = scores[:, np.newaxis]

        scale = 1.0

        # バウンディングボックス取得
        bboxes = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 0:2] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 2:4] * np.exp(loc[:, 2:4] * self.VARIANCE)) *
             scale))
        bboxes[:, 0:2] -= bboxes[:, 2:4] / 2

        # ランドマーク取得
        landmarks = np.hstack(
            ((self.priors[:, 0:2] +
              loc[:, 4:6] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 6:8] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 8:10] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 10:12] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale,
             (self.priors[:, 0:2] +
              loc[:, 12:14] * self.VARIANCE[0] * self.priors[:, 2:4]) * scale))

        dets = np.hstack((bboxes, landmarks, scores))

        return dets

@cv.gapi.op('custom.PreProcess',
            in_types=[cv.GMat],
            out_types=[cv.GMat])
class PreProcess:
    @staticmethod
    def outMeta(image):
        return cv.GMatDesc(-1, -1, (-1, -1))

@cv.gapi.kernel(PreProcess)
class PreProcessImpl:
    @staticmethod
    def run(image):
        # BGR -> RGB 変換
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # # リサイズ
        # image = cv.resize(
        #     image,
        #     (160, 120),
        #     interpolation=cv.INTER_LINEAR,
        # )

        # # リシェイプ
        # image = image.astype(np.float32)
        # image = image.transpose((2, 0, 1))
        # image = image.reshape(1, 3, 120, 160)

        return image


@cv.gapi.op('custom.PostProcess',
            in_types=[cv.GMat, cv.GMat, cv.GMat, cv.GMat],
            out_types=[cv.GArray.Rect])
class PostProcess:
    @staticmethod
    def outMeta(image, loc, conf, iou):
        return cv.empty_array_desc() # cv.GMatDesc(-1, -1, (-1, -1)), cv.GMatDesc(-1, -1, (-1, -1)), cv.GMatDesc(-1, -1, (-1, -1))

@cv.gapi.kernel(PostProcess)
class PostProcessImpl:
    @staticmethod
    def setup(image, loc, conf, iou):
        return YuNetONNX(conf_th=0.6)

    @staticmethod
    def run(image, loc, conf, iou, status):
        dets = status.postprocess((loc, conf, iou))
        # import IPython; IPython.embed()
        bboxes, landmarks, scores = np.array(dets[0]), np.array(dets[1]), np.array(dets[2])
        if len(bboxes) == 0:
            return []
        # import IPython; IPython.embed()
        bboxes[:, [0,2]] *= image.shape[1]
        bboxes[:, [1,3]] *= image.shape[0]

        # import IPython; IPython.embed()
        return bboxes.astype(np.int32) # , landmarks.astype(np.int32), scores

def process(g_in):
    face_inputs = cv.GInferInputs()
    face_inputs.setInput('input', PreProcess.on(g_in))
    face_outputs = cv.gapi.infer('face-detection', face_inputs)
    loc = face_outputs.at('loc')
    conf = face_outputs.at('conf')
    iou = face_outputs.at('iou')
    bbox = PostProcess.on(g_in, loc, conf, iou)
    return bbox

def kernels():
    return [PostProcessImpl, PreProcessImpl]

def networks():
    return [cv.gapi.onnx.params(
        'face-detection', "models/face_detection_yunet_120x160.onnx",
        ["CUDAExecutionProvider"]).cfgNormalize([False])]