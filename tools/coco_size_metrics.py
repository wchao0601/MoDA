"""COCO-style AP by object area for the project's native detection validator."""

from __future__ import annotations

from collections import defaultdict
import multiprocessing as mp

import numpy as np
import torch

from ultralytics.models.yolo.obb.val import OBBValidator
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import batch_probiou, box_iou


AREA_RANGES = {
    "small": (0.0, 32.0**2),
    "medium": (32.0**2, 96.0**2),
    "large": (96.0**2, float("inf")),
}
IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
_PARALLEL_RECORDS = None


class CocoSizeValidator(OBBValidator):
    """Collect native-scale detections and calculate COCO-style APs/m/l."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.area_records = []

    def update_metrics(self, preds, batch):
        # OBBValidator._prepare_batch scales boxes in place. Collect from a cloned
        # bbox tensor first so the native validator receives its untouched batch.
        area_batch = {**batch, "bboxes": batch["bboxes"].clone()}
        self._collect_area_records(preds, area_batch)
        super().update_metrics(preds, batch)

    def _collect_area_records(self, preds, batch):
        for image_index, prediction in enumerate(preds):
            prepared_batch = self._prepare_batch(image_index, batch)
            gt_classes = prepared_batch["cls"].detach().cpu().numpy().astype(int)
            gt_boxes = prepared_batch["bbox"].detach().cpu().numpy()

            if len(prediction):
                native_prediction = self._prepare_pred(prediction, prepared_batch)
                pred_boxes = torch.cat(
                    [native_prediction[:, :4], native_prediction[:, -1:]], dim=-1
                ).detach().cpu().numpy()
                pred_scores = native_prediction[:, 4].detach().cpu().numpy()
                pred_classes = native_prediction[:, 5].detach().cpu().numpy().astype(int)
            else:
                pred_boxes = np.empty((0, 5), dtype=np.float32)
                pred_scores = np.empty(0, dtype=np.float32)
                pred_classes = np.empty(0, dtype=int)

            self.area_records.append(
                {
                    "gt_boxes": gt_boxes,
                    "gt_classes": gt_classes,
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "pred_classes": pred_classes,
                }
            )

    def get_stats(self):
        stats = super().get_stats()
        size_metrics = evaluate_size_ap(self.area_records, self.nc)
        self.metrics.size_ap = size_metrics
        return {**stats, **{f"metrics/AP_{key}(B)": value for key, value in size_metrics.items()}}


class CocoSizeDetectionValidator(DetectionValidator):
    """Collect axis-aligned detections and calculate COCO-style APs/m/l."""

    def init_metrics(self, model):
        super().init_metrics(model)
        self.area_records = []

    def update_metrics(self, preds, batch):
        area_batch = {**batch, "bboxes": batch["bboxes"].clone()}
        for image_index, prediction in enumerate(preds):
            prepared_batch = self._prepare_batch(image_index, area_batch)
            gt_classes = prepared_batch["cls"].detach().cpu().numpy().astype(int)
            gt_boxes = prepared_batch["bbox"].detach().cpu().numpy()
            if len(prediction):
                native_prediction = self._prepare_pred(prediction, prepared_batch)
                pred_boxes = native_prediction[:, :4].detach().cpu().numpy()
                pred_scores = native_prediction[:, 4].detach().cpu().numpy()
                pred_classes = native_prediction[:, 5].detach().cpu().numpy().astype(int)
            else:
                pred_boxes = np.empty((0, 4), dtype=np.float32)
                pred_scores = np.empty(0, dtype=np.float32)
                pred_classes = np.empty(0, dtype=int)
            self.area_records.append(
                {"gt_boxes": gt_boxes, "gt_classes": gt_classes, "pred_boxes": pred_boxes,
                 "pred_scores": pred_scores, "pred_classes": pred_classes}
            )
        super().update_metrics(preds, batch)

    def get_stats(self):
        stats = super().get_stats()
        size_metrics = evaluate_size_ap(self.area_records, self.nc)
        self.metrics.size_ap = size_metrics
        return {**stats, **{f"metrics/AP_{key}(B)": value for key, value in size_metrics.items()}}


def box_area(boxes: np.ndarray) -> np.ndarray:
    """Return xywhr or xyxy box areas in original-image pixels."""
    if not len(boxes):
        return np.empty(0, dtype=np.float32)
    if boxes.shape[1] == 5:
        return np.maximum(0.0, boxes[:, 2]) * np.maximum(0.0, boxes[:, 3])
    return np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])


def pairwise_iou(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    if not len(gt_boxes) or not len(pred_boxes):
        return np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float32)
    gt_tensor, pred_tensor = torch.from_numpy(gt_boxes), torch.from_numpy(pred_boxes)
    if gt_boxes.shape[1] == 5:
        return batch_probiou(gt_tensor, pred_tensor).numpy()
    return box_iou(gt_tensor, pred_tensor).numpy()


def evaluate_size_ap(records: list[dict], num_classes: int) -> dict[str, float]:
    """Calculate AP@[.50:.95] for COCO small, medium, and large ranges."""
    # IoU depends on neither class nor area range.  Compute it once per image
    # instead of repeating it for every class x size evaluation cell.
    for record in records:
        if "ious" not in record:
            record["ious"] = pairwise_iou(record["gt_boxes"], record["pred_boxes"])
    tasks = [(name, class_id, area_range) for name, area_range in AREA_RANGES.items() for class_id in range(num_classes)]
    if len(records) >= 5000 and "fork" in mp.get_all_start_methods():
        global _PARALLEL_RECORDS
        _PARALLEL_RECORDS = records
        with mp.get_context("fork").Pool(processes=min(8, len(tasks)), initializer=_size_worker_init) as pool:
            values = pool.map(_parallel_class_ap, tasks)
        _PARALLEL_RECORDS = None
    else:
        values = [(name, evaluate_class_area(records, class_id, area_range)) for name, class_id, area_range in tasks]
    grouped = defaultdict(list)
    for area_name, class_ap in values:
        if class_ap is not None:
            grouped[area_name].append(class_ap)
    return {name: float(np.mean(grouped[name])) if grouped[name] else float("nan") for name in AREA_RANGES}


def _size_worker_init() -> None:
    """Keep each forked worker single-threaded; tasks are parallel across processes."""
    torch.set_num_threads(1)


def _parallel_class_ap(task: tuple[str, int, tuple[float, float]]) -> tuple[str, float | None]:
    area_name, class_id, area_range = task
    return area_name, evaluate_class_area(_PARALLEL_RECORDS, class_id, area_range)


def evaluate_area(
    records: list[dict],
    num_classes: int,
    area_range: tuple[float, float],
) -> float:
    class_aps = [evaluate_class_area(records, class_id, area_range) for class_id in range(num_classes)]
    class_aps = [value for value in class_aps if value is not None]
    return float(np.mean(class_aps)) if class_aps else float("nan")


def evaluate_class_area(records: list[dict], class_id: int, area_range: tuple[float, float]) -> float | None:
    """Calculate one class/area cell; cells are independent and safe to parallelize."""
    detections = defaultdict(list)
    ground_truth_count = 0

    for record in records:
        gt_mask = record["gt_classes"] == class_id
        pred_mask = record["pred_classes"] == class_id
        gt_boxes = record["gt_boxes"][gt_mask]
        pred_boxes = record["pred_boxes"][pred_mask]
        pred_scores = record["pred_scores"][pred_mask]
        gt_ignored = ~area_mask(box_area(gt_boxes), area_range)
        pred_areas = box_area(pred_boxes)
        ground_truth_count += int((~gt_ignored).sum())

        order = np.argsort(-pred_scores, kind="stable")[:100]
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]
        pred_areas = pred_areas[order]
        ious = record["ious"][gt_mask][:, pred_mask][:, order]

        # COCO sorts ignored GT last, so valid objects receive matching priority.
        gt_order = np.argsort(gt_ignored, kind="stable")
        gt_ignored = gt_ignored[gt_order]
        ious = ious[gt_order]

        # COCO matching is independent at every IoU threshold.  Track all ten
        # thresholds in one matrix to avoid Python loops over thresholds and GT.
        gt_matches = np.zeros((len(IOU_THRESHOLDS), len(gt_boxes)), dtype=bool)
        threshold_rows = np.arange(len(IOU_THRESHOLDS))
        pred_inside_area = area_mask(pred_areas, area_range)
        for prediction_index, score in enumerate(pred_scores):
            if len(gt_boxes):
                eligible = (~gt_matches) & (ious[:, prediction_index][None, :] >= IOU_THRESHOLDS[:, None])
                valid_eligible = eligible & (~gt_ignored[None, :])
                has_valid = valid_eligible.any(axis=1)
                eligible[has_valid] = valid_eligible[has_valid]
                candidate_iou = np.where(eligible, ious[:, prediction_index][None, :], -1.0)
                best_gt = candidate_iou.argmax(axis=1)
                matched = candidate_iou[threshold_rows, best_gt] >= 0.0
                gt_matches[threshold_rows[matched], best_gt[matched]] = True
                matched_ignored = matched & gt_ignored[best_gt]
            else:
                matched = np.zeros(len(IOU_THRESHOLDS), dtype=bool)
                matched_ignored = matched
            unmatched_outside_area = (~matched) & (~pred_inside_area[prediction_index])
            for threshold_index in threshold_rows:
                detections[int(threshold_index)].append(
                    (float(score), bool(matched[threshold_index] and not matched_ignored[threshold_index]),
                     bool(matched_ignored[threshold_index] or unmatched_outside_area[threshold_index]))
                )

    if ground_truth_count == 0:
        return None
    threshold_aps = [interpolated_ap(detections[index], ground_truth_count) for index in range(len(IOU_THRESHOLDS))]
    return float(np.mean(threshold_aps))


def area_mask(areas: np.ndarray, area_range: tuple[float, float]) -> np.ndarray:
    minimum, maximum = area_range
    return (areas >= minimum) & (areas < maximum)


def interpolated_ap(detections: list[tuple[float, bool, bool]], num_gt: int) -> float:
    """COCO 101-point interpolated AP for one class and IoU threshold."""
    detections.sort(key=lambda item: item[0], reverse=True)
    valid = [(is_tp, ignored) for _, is_tp, ignored in detections if not ignored]
    if not valid:
        return 0.0

    true_positives = np.cumsum([is_tp for is_tp, _ in valid], dtype=float)
    false_positives = np.cumsum([not is_tp for is_tp, _ in valid], dtype=float)
    recall = true_positives / max(num_gt, 1)
    precision = true_positives / np.maximum(true_positives + false_positives, 1e-16)
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    sampled = np.zeros_like(RECALL_THRESHOLDS)
    indices = np.searchsorted(recall, RECALL_THRESHOLDS, side="left")
    valid_indices = indices < len(precision)
    sampled[valid_indices] = precision[indices[valid_indices]]
    return float(sampled.mean())
