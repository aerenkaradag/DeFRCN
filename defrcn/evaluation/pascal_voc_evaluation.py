# -*- coding: utf-8 -*-
from itertools import tee
import os
import torch
import logging
import tempfile
import numpy as np
from functools import lru_cache
from xml.etree import ElementTree as ET
from collections import OrderedDict, defaultdict
from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import create_small_table
from defrcn.evaluation.evaluator import DatasetEvaluator


class PascalVOCDetectionEvaluator(DatasetEvaluator):
    """
    Evaluate Pascal VOC AP.
    It contains a synchronization, therefore has to be called from all ranks.

    Note that this is a rewrite of the official Matlab API.
    The results should be similar, but not identical to the one produced by
    the official API.
    """

    def __init__(self, dataset_name):
        """
        Args:
            dataset_name (str): name of the dataset, e.g., "voc_2007_test"
        """
        self._dataset_name = dataset_name
        meta = MetadataCatalog.get(dataset_name)
        self._anno_file_template = os.path.join(meta.dirname, "Annotations", "{}.xml")
        self._image_set_path = os.path.join(meta.dirname, "ImageSets", "Main", meta.split + ".txt")
        self._class_names = meta.thing_classes
        # add this two terms for calculating the mAP of different subset
        self._base_classes = meta.base_classes
        self._novel_classes = meta.novel_classes
        assert meta.year in [2007, 2012], meta.year
        self._is_2007 = meta.year == 2007
        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

    def reset(self):
        self._predictions = defaultdict(list)  # class name -> list of prediction strings

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            image_id = input["image_id"]
            instances = output["instances"].to(self._cpu_device)
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.tolist()
            classes = instances.pred_classes.tolist()            
            for box, score, cls in zip(boxes, scores, classes):
                #if score > 0.5:
                #print("box, score, cls: ", box, score, cls)
                xmin, ymin, xmax, ymax = box
                # The inverse of data loading logic in `datasets/pascal_voc.py`
                xmin += 1
                ymin += 1
                self._predictions[cls].append(
                    f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
                )
                #print(f"{image_id} {score:.3f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}")

    def evaluate(self):
        """
        Returns:
            dict: has a key "segm", whose value is a dict of "AP", "AP50", and "AP75".
        """
        all_predictions = comm.gather(self._predictions, dst=0)
        if not comm.is_main_process():
            return
        predictions = defaultdict(list)
        for predictions_per_rank in all_predictions:
            for clsid, lines in predictions_per_rank.items():
                #print("clsid, lines: ", clsid, lines)
                predictions[clsid].extend(lines)
        del all_predictions

        self._logger.info(
            "Evaluating {} using {} metric. "
            "Note that results do not use the official Matlab API.".format(
                self._dataset_name, 2007 if self._is_2007 else 2012
            )
        )

        with tempfile.TemporaryDirectory(prefix="pascal_voc_eval_") as dirname:
            res_file_template = os.path.join(dirname, "{}.txt")

            aps = defaultdict(list)  # iou -> ap per class
            aps_base = defaultdict(list)
            aps_novel = defaultdict(list)
            exist_base, exist_novel = False, False

            with open("/root/DeFRCN/output.txt", "w") as output_file:
                for cls_id, lines in predictions.items():
                    # Write class identifier and class name to the file
                    output_file.write(f"Class ID: {cls_id}, Class Name: {self._class_names[cls_id]}\n")
                    # Write each line of predictions to the file
                    for line in lines:
                        output_file.write(f"{line}\n")

            with open("/root/DeFRCN/sortedinput.txt", "w") as output_file:
                for cls_id, lines in predictions.items():
                    splitlines = [x.strip().split(" ") for x in lines]
                    image_ids = [x[0] for x in splitlines]
                    confidence = np.array([float(x[1]) for x in splitlines])
                    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

                    # sort by confidence
                    sorted_ind = np.argsort(-confidence)
                    sorted_confidences = confidence[sorted_ind]
                    #print("sorted_ind: ", sorted_ind)
                    BB = BB[sorted_ind, :]
                    image_ids = [image_ids[x] for x in sorted_ind]
                    # Create a dictionary to store image_id, bbox, and confidence
                    id_dict = {image_id: [] for image_id in image_ids}

                    for i in range(len(image_ids)):
                        id_dict[image_ids[i]].append((BB[i], sorted_confidences[i]))

                    output_directory = "/root/DeFRCN/bilgilendirici2/"
                    os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist

                    with open(os.path.join(output_directory, f"{self._class_names[cls_id]}.txt"), "w") as f:
                        # Write each line of predictions to the file
                        for image_id, values in id_dict.items():
                            for value in values:
                                bbox, conf = value
                                f.write(f"{image_id} {bbox} {conf}\n")
            print("******************")
            for cls_id, cls_name in enumerate(self._class_names):
                print("cls_id, cls_name: ", cls_id, cls_name)
                lines = predictions.get(cls_id, [""])

                with open(res_file_template.format(cls_name), "w") as f:
                    f.write("\n".join(lines))

                for thresh in range(50, 100, 5):
                    rec, prec, ap = voc_eval(
                        res_file_template,
                        self._anno_file_template,
                        self._image_set_path,
                        cls_name,
                        ovthresh=thresh / 100.0,
                        use_07_metric=self._is_2007,
                    )
                    aps[thresh].append(ap * 100)

                    if self._base_classes is not None and cls_name in self._base_classes:
                        aps_base[thresh].append(ap * 100)
                        exist_base = True

                    if self._novel_classes is not None and cls_name in self._novel_classes:
                        aps_novel[thresh].append(ap * 100)
                        exist_novel = True
            # Copy the temporary directory to another location before it gets deleted
            import shutil
            destination_path = "/root/DeFRCN/hedef/"
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
            shutil.copytree(dirname, destination_path)

        ret = OrderedDict()
        mAP = {iou: np.mean(x) for iou, x in aps.items()}
        ret["bbox"] = {"AP": np.mean(list(mAP.values())), "AP50": mAP[50], "AP75": mAP[75]}

        # adding evaluation of the base and novel classes
        if exist_base:
            mAP_base = {iou: np.mean(x) for iou, x in aps_base.items()}
            ret["bbox"].update(
                {"bAP": np.mean(list(mAP_base.values())), "bAP50": mAP_base[50],
                 "bAP75": mAP_base[75]}
            )

        if exist_novel:
            mAP_novel = {iou: np.mean(x) for iou, x in aps_novel.items()}
            ret["bbox"].update({
                "nAP": np.mean(list(mAP_novel.values())), "nAP50": mAP_novel[50],
                "nAP75": mAP_novel[75]
            })

        # write per class AP to logger
        per_class_res = {self._class_names[idx]: ap for idx, ap in enumerate(aps[50])}

        self._logger.info("Evaluate per-class mAP50:\n"+create_small_table(per_class_res))
        self._logger.info("Evaluate overall bbox:\n"+create_small_table(ret["bbox"]))
        return ret


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        objects.append(obj_struct)

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name

    # first load gt
    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    # load annots
    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        # difficult = np.array([False for x in R]).astype(np.bool)  # treat all "difficult" as GT
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "difficult": difficult, "det": det}
        #print(f"bbox: {bbox}, difficult: {difficult}, det: {det}") # henüz difficultlar ve detler False

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]).reshape(-1, 4)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_confidences = confidence[sorted_ind]
    #print("sorted_ind: ", sorted_ind)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    

    # go down dets and mark TPs and FPs
    nd = len(image_ids) # nd=16845, 200 frame için, her detectionı sayıyor
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    output_directory = "/root/DeFRCN/bilgilendirici/"
    os.makedirs(output_directory, exist_ok=True)  # Create the output directory if it doesn't exist
    with open(os.path.join(output_directory, f"{classname}.txt"), "w") as f:
    #with open(os.path.join(output_directory, f"ballet.txt"), "w") as f:
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float) # bb modelin bulduğu bbox
            ovmax = -np.inf
            BBGT = R["bbox"].astype(float) # BBGT gerçek bbox
            #print(f"image_ids: {image_ids[d]}, bbox: {bb}, gt: {BBGT}, confidence: {sorted_confidences[d]}")
            f.write(f"image_ids: {image_ids[d]}, bbox: {bb}, gt: {BBGT}, confidence: {sorted_confidences[d]}\n")

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                ih = np.maximum(iymax - iymin + 1.0, 0.0)
                inters = iw * ih

                # union
                uni = (
                    (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                    + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                    - inters
                )

                overlaps = inters / uni
                ovmax = np.max(overlaps) # o framedeki max IOU değeri
                jmax = np.argmax(overlaps) # o max değerin sırası(aynı frame numberdaki confidence'e göre listelenmişler arasındaki sıra)

            if ovmax > ovthresh:
                if not R["difficult"][jmax]: # bu condition hep sağlanıyor
                    if not R["det"][jmax]: # henüz tespit edilmemişse giriyor
                        tp[d] = 1.0
                        R["det"][jmax] = 1 # artık d. satırda box detected ve bu TP için +1  
                    else:
                        fp[d] = 1.0 # undetected, FP
            else:
                fp[d] = 1.0 # thresholdu geçemediyse FP

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    print("fp: ", fp)
    print("tp: ", tp)
    print("rec: ", rec)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    print("prec: ", prec)
    ap = voc_ap(rec, prec, use_07_metric)
    print("ap: ", ap)

    return rec, prec, ap
