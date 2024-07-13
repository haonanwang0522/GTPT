# # ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict
from collections import OrderedDict
import logging
import os

from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import json_tricks as json
import numpy as np

from dataset.JointsDataset import JointsDataset
from nms.nms import oks_nms
from nms.nms import soft_oks_nms


logger = logging.getLogger(__name__)


class COCOWholeBodyDataset(JointsDataset):
    '''
    COCO-WholeBody keypoints::

        0-16: 17 body keypoints,
        17-22: 6 foot keypoints,
        23-90: 68 face keypoints,
        91-132: 42 hand keypoints

        In total, we have 133 keypoints for wholebody pose estimation.,
	"skeleton": [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    '''
    def __init__(self, cfg, root, image_set, is_train, transform=None, coord_representation='heatmap', simdr_split_ratio=1):
        super().__init__(cfg, root, image_set, is_train, transform, coord_representation, simdr_split_ratio)
        self.nms_thre = cfg.TEST.NMS_THRE
        self.image_thre = cfg.TEST.IMAGE_THRE
        self.soft_nms = cfg.TEST.SOFT_NMS
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.bbox_file = cfg.TEST.COCO_BBOX_FILE
        self.use_gt_bbox = cfg.TEST.USE_GT_BBOX
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200

        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict(
            [
                (self._class_to_coco_ind[cls], self._class_to_ind[cls])
                for cls in self.classes[1:]
            ]
        )

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)

        logger.info('=> num_images: {}'.format(self.num_images))

        self.num_joints = 133
        self.body_num = 17
        self.foot_num = 6
        self.face_num = 68
        self.left_hand_num = 21
        self.right_hand_num = 21
        body_flip_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]
        foot_flip_pairs = [[17, 20], [18, 21], [19, 22]]
        face_flip_pairs = [
            [23, 39], [24, 38], [25, 37], [26, 36], [27, 35], [28, 34], [29, 33], [30, 32],
            [40, 49], [41, 48], [42, 47], [43, 46], [44, 45], 
            [54, 58], [55, 57], 
            [59, 68], [60, 67], [61, 66], [62, 65], [63, 70], [64, 69], 
            [71, 77], [72, 76], [73, 75],
            [78, 82], [79, 81], [83, 87], [84, 86], [88, 90],
        ]
        hand_flip_pairs = [
            [91, 112], [92, 113], [93, 114], [94, 115], [95, 116],
            [96, 117], [97, 118], [98, 119], [99, 120],
            [100, 121], [101, 122], [102, 123], [103, 124],
            [104, 125], [105, 126], [106, 127], [107, 128],
            [108, 129], [109, 130], [110, 131], [111, 132],
        ]
        self.sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.068, 0.066, 0.066,
            0.092, 0.094, 0.094, 0.042, 0.043, 0.044, 0.043, 0.040, 0.035, 0.031,
            0.025, 0.020, 0.023, 0.029, 0.032, 0.037, 0.038, 0.043, 0.041, 0.045,
            0.013, 0.012, 0.011, 0.011, 0.012, 0.012, 0.011, 0.011, 0.013, 0.015,
            0.009, 0.007, 0.007, 0.007, 0.012, 0.009, 0.008, 0.016, 0.010, 0.017,
            0.011, 0.009, 0.011, 0.009, 0.007, 0.013, 0.008, 0.011, 0.012, 0.010,
            0.034, 0.008, 0.008, 0.009, 0.008, 0.008, 0.007, 0.010, 0.008, 0.009,
            0.009, 0.009, 0.007, 0.007, 0.008, 0.011, 0.008, 0.008, 0.008, 0.01,
            0.008, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024, 0.035,
            0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02, 0.019,
            0.022, 0.031, 0.029, 0.022, 0.035, 0.037, 0.047, 0.026, 0.025, 0.024,
            0.035, 0.018, 0.024, 0.022, 0.026, 0.017, 0.021, 0.021, 0.032, 0.02,
            0.019, 0.022, 0.031
        ])

        self.flip_pairs = body_flip_pairs + foot_flip_pairs + face_flip_pairs + hand_flip_pairs
        self.parent_ids = None
        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10) + tuple(range(23,133))
        self.lower_body_ids = (11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)

        self.joints_weight = np.array(
            [1.] * self.num_joints,
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_ann_file_keypoint(self):
        """ self.root / annotations / person_keypoints_train2017.json """
        prefix = 'coco_wholebody' \
            if 'test' not in self.image_set else 'image_info'
        return os.path.join(
            self.root,
            'annotations',
            prefix + '_' + self.image_set + '_v1.0.json'
        )

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        if self.is_train or self.use_gt_bbox:
            # use ground truth bbox
            print("++++++++++++++++++++++++++++++++++++++++++++")
            gt_db = self._load_coco_keypoint_annotations()
        else:
            # use bbox from detection
            print("=======================================")
            gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            ann = obj
            _keypoints = ann['keypoints'] + ann['foot_kpts'] + \
                         ann['face_kpts'] + ann['lefthand_kpts'] + \
                         ann['righthand_kpts']
            _keypoints = np.array(_keypoints).reshape(-1, 3)
            k = _keypoints.shape[0]
            # joints_3d = _keypoints.copy
            # joints_3d[:, 2] = 0
            # joints_3d_vis = np.minimum(1, _keypoints[..., 2:] > 0)

            joints_3d = np.zeros((k, 3), dtype=np.float)
            joints_3d_vis = np.zeros((k, 3), dtype=np.float)
            for ipt in range(k):
                joints_3d[ipt, 0] = _keypoints[ipt, 0]
                joints_3d[ipt, 1] = _keypoints[ipt, 1]
                joints_3d[ipt, 2] = 0
                t_vis = _keypoints[ipt, 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: images / train2017 / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.image_set:
            file_name = 'COCO_%s_' % self.image_set + file_name

        prefix = 'test2017' if 'test' in self.image_set else self.image_set

        data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix

        image_path = os.path.join(
            self.root, 'images', data_name, file_name)

        return image_path

    def _load_coco_person_detection_results(self):
        all_boxes = None
        with open(self.bbox_file, 'r') as f:
            all_boxes = json.load(f)

        if not all_boxes:
            logger.error('=> Load %s fail!' % self.bbox_file)
            return None

        logger.info('=> Total boxes: {}'.format(len(all_boxes)))

        kpt_db = []
        num_boxes = 0
        for n_img in range(0, len(all_boxes)):
            det_res = all_boxes[n_img]
            if det_res['category_id'] != 1:
                continue
            img_name = self.image_path_from_index(det_res['image_id'])
            box = det_res['bbox']
            score = det_res['score']

            if score < self.image_thre:
                continue

            num_boxes = num_boxes + 1

            center, scale = self._box2cs(box)
            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.ones(
                (self.num_joints, 3), dtype=np.float)
            kpt_db.append({
                'image': img_name,
                'center': center,
                'scale': scale,
                'score': score,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
            })

        logger.info('=> Total boxes after fliter low score@{}: {}'.format(
            self.image_thre, num_boxes))
        return kpt_db

    def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
                 *args, **kwargs):
        rank = cfg.RANK

        res_folder = os.path.join(output_dir, 'results')
        if not os.path.exists(res_folder):
            try:
                os.makedirs(res_folder)
            except Exception:
                logger.error('Fail to make {}'.format(res_folder))

        res_file = os.path.join(
            res_folder, 'keypoints_{}_results_{}.json'.format(
                self.image_set, rank)
        )

        # person x (keypoints)

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num, self.left_hand_num,
            self.right_hand_num
        ])

        _kpts = []
        for idx, kpt in enumerate(preds):
            _kpts.append({
                'keypoints': kpt[cuts[0]:cuts[1]],
                'foot_kpts': kpt[cuts[1]:cuts[2]],
                'face_kpts': kpt[cuts[2]:cuts[3]],
                'lefthand_kpts': kpt[cuts[3]:cuts[4]],
                'righthand_kpts': kpt[cuts[4]:cuts[5]],
                'wholebody_kpts': kpt,
                'center': all_boxes[idx][0:2],
                'scale': all_boxes[idx][2:4],
                'area': all_boxes[idx][4],
                'score': all_boxes[idx][5],
                'image': int(img_path[idx][-16:-4])
            })
        # image x person x (keypoints)
        kpts = defaultdict(list)
        for kpt in _kpts:
            kpts[kpt['image']].append(kpt)

        # rescoring and oks nms
        num_joints = self.num_joints
        in_vis_thre = self.in_vis_thre
        oks_thre = self.oks_thre
        oks_nmsed_kpts = []
        for img in kpts.keys():
            img_kpts = kpts[img]

            for n_p in img_kpts:
                box_score = n_p['score']
                kpt_score = 0
                valid_num = 0
                for n_jt in range(0, num_joints):
                    t_s = n_p['wholebody_kpts'][n_jt][2]
                    if t_s > in_vis_thre:
                        kpt_score = kpt_score + t_s
                        valid_num = valid_num + 1
                if valid_num != 0:
                    kpt_score = kpt_score / valid_num
                # rescoring
                n_p['score'] = kpt_score * box_score

            if self.soft_nms:
                keep = soft_oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre, self.sigmas
                )
            else:
                # default
                keep = oks_nms(
                    [img_kpts[i] for i in range(len(img_kpts))],
                    oks_thre, self.sigmas
                )

            if len(keep) == 0:
                oks_nmsed_kpts.append(img_kpts)
            else:
                oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

        self._write_coco_keypoint_results(
            oks_nmsed_kpts, res_file)
        if 'test' not in self.image_set:
            info_str = self._do_python_keypoint_eval(
                res_file, res_folder)
            name_value = OrderedDict(info_str)
            return name_value, name_value['AP']
        else:
            return {'Null': 0}, 0

    def _write_coco_keypoint_results(self, keypoints, res_file):
        data_pack = [
            {
                'cat_id': self._class_to_coco_ind[cls],
                'cls_ind': cls_ind,
                'cls': cls,
                'ann_type': 'keypoints',
                'keypoints': keypoints
            }
            for cls_ind, cls in enumerate(self.classes) if not cls == '__background__'
        ]

        results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
        logger.info('=> writing results json to %s' % res_file)
        with open(res_file, 'w') as f:
            json.dump(results, f, sort_keys=True, indent=4)
        try:
            json.load(open(res_file))
        except Exception:
            content = []
            with open(res_file, 'r') as f:
                for line in f:
                    content.append(line)
            content[-1] = ']'
            with open(res_file, 'w') as f:
                for c in content:
                    f.write(c)

    def _coco_keypoint_results_one_category_kernel(self, data_pack):
        cat_id = data_pack['cat_id']
        keypoints = data_pack['keypoints']
        cat_results = []

        for img_kpts in keypoints:
            B = len(img_kpts)
            if B == 0:
                continue
            
            # _key_points = np.array([img_kpts[k]['keypoints']
            #                         for k in range(len(img_kpts))])
            # _key_points = _key_points.reshape(B, -1)
            result = [
                {
                    'image_id': img_kpts[k]['image'],
                    'category_id': cat_id,
                    'keypoints': img_kpts[k]['keypoints'].flatten().tolist(),
                    'foot_kpts': img_kpts[k]['foot_kpts'].flatten().tolist(),
                    'face_kpts': img_kpts[k]['face_kpts'].flatten().tolist(),
                    'lefthand_kpts': img_kpts[k]['lefthand_kpts'].flatten().tolist(),
                    'righthand_kpts': img_kpts[k]['righthand_kpts'].flatten().tolist(),
                    'score': img_kpts[k]['score'],
                    'center': list(img_kpts[k]['center']),
                    'scale': list(img_kpts[k]['scale'])
                }
                for k in range(B)
            ]
            cat_results.extend(result)

        return cat_results

    def _do_python_keypoint_eval(self, res_file, res_folder):
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.sigmas

        cuts = np.cumsum([
            0, self.body_num, self.foot_num, self.face_num, self.left_hand_num,
            self.right_hand_num
        ])

        info_str = []
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

        print('*'*20 + " evaluate body")
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_body',
            sigmas[cuts[0]:cuts[1]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for ind, name in enumerate(stats_names):
            info_str.append(('body_'+name, coco_eval.stats[ind]))

        print('*'*20 + " evaluate foot")
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_foot',
            sigmas[cuts[1]:cuts[2]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for ind, name in enumerate(stats_names):
            info_str.append(('foot_'+name, coco_eval.stats[ind]))

        print('*'*20 + " evaluate face")
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_face',
            sigmas[cuts[2]:cuts[3]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for ind, name in enumerate(stats_names):
            info_str.append(('face_'+name, coco_eval.stats[ind]))

        print('*'*20 + " evaluate left hand")
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_lefthand',
            sigmas[cuts[3]:cuts[4]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for ind, name in enumerate(stats_names):
            info_str.append(('lefthand_'+name, coco_eval.stats[ind]))

        print('*'*20 + " evaluate right hand")
        coco_eval = COCOeval(
            self.coco,
            coco_det,
            'keypoints_righthand',
            sigmas[cuts[4]:cuts[5]],
            use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        for ind, name in enumerate(stats_names):
            info_str.append(('righthand_'+name, coco_eval.stats[ind]))

        print('*'*20 + " evaluate whole body")
        coco_eval = COCOeval(
            self.coco, coco_det, 'keypoints_wholebody', sigmas, use_area=True)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))

        return info_str
