"""Evaluate tracking CSV output against MOT17 ground truth."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


IOU_THRESHOLD = 0.5
PREDICTION_FIELDS = [
    "frame_id",
    "person_id",
    "bbox_center_x",
    "bbox_center_y",
    "bbox_width",
    "bbox_height",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate tracking results against MOT17 ground truth"
    )
    parser.add_argument("--gt", required=True, help="Path to the MOT17 gt.txt file")
    parser.add_argument("--pred", required=True, help="Path to the prediction CSV")
    parser.add_argument(
        "--max-frame",
        type=int,
        default=None,
        help="Only evaluate frames <= max-frame",
    )
    return parser.parse_args()


def load_ground_truth(gt_path, max_frame=None):
    """Load MOT17 frame, ID, and bounding-box fields, ignoring other columns."""
    ground_truth = defaultdict(list)

    with Path(gt_path).open("r", newline="", encoding="utf-8-sig") as gt_file:
        reader = csv.reader(gt_file)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"Invalid GT row {row_number}: expected at least 6 fields")

            try:
                frame_id = int(row[0])
                if max_frame is not None and frame_id > max_frame:
                    continue
                gt_id = int(row[1])
                x, y, width, height = map(float, row[2:6])
            except ValueError as error:
                raise ValueError(f"Invalid GT values on row {row_number}") from error

            ground_truth[frame_id].append(
                (gt_id, np.array([x, y, x + width, y + height], dtype=float))
            )

    return ground_truth


def load_predictions(pred_path, max_frame=None):
    """Load prediction rows and convert center-format boxes to corner format."""
    predictions = defaultdict(list)

    with Path(pred_path).open("r", newline="", encoding="utf-8-sig") as pred_file:
        reader = csv.DictReader(pred_file)
        if reader.fieldnames != PREDICTION_FIELDS:
            raise ValueError(
                "Prediction CSV header must be: " + ",".join(PREDICTION_FIELDS)
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                frame_id = int(row["frame_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid prediction values on row {row_number}") from error

            if max_frame is not None and frame_id > max_frame:
                continue

            person_id_text = row["person_id"].strip()
            if not person_id_text:
                # TrackingLogger writes an otherwise empty row for frames with no targets.
                continue

            try:
                person_id = int(person_id_text)
                center_x = float(row["bbox_center_x"])
                center_y = float(row["bbox_center_y"])
                width = float(row["bbox_width"])
                height = float(row["bbox_height"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid prediction values on row {row_number}") from error

            half_width = width / 2.0
            half_height = height / 2.0
            predictions[frame_id].append(
                (
                    person_id,
                    np.array(
                        [
                            center_x - half_width,
                            center_y - half_height,
                            center_x + half_width,
                            center_y + half_height,
                        ],
                        dtype=float,
                    ),
                )
            )

    return predictions


def calculate_iou_matrix(gt_boxes, pred_boxes):
    """Calculate pairwise IoU values for two arrays of corner-format boxes."""
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return np.empty((len(gt_boxes), len(pred_boxes)), dtype=float)

    gt_boxes = np.asarray(gt_boxes, dtype=float)
    pred_boxes = np.asarray(pred_boxes, dtype=float)

    intersection_x1 = np.maximum(gt_boxes[:, None, 0], pred_boxes[None, :, 0])
    intersection_y1 = np.maximum(gt_boxes[:, None, 1], pred_boxes[None, :, 1])
    intersection_x2 = np.minimum(gt_boxes[:, None, 2], pred_boxes[None, :, 2])
    intersection_y2 = np.minimum(gt_boxes[:, None, 3], pred_boxes[None, :, 3])

    intersection_width = np.maximum(0.0, intersection_x2 - intersection_x1)
    intersection_height = np.maximum(0.0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    gt_area = np.maximum(0.0, gt_boxes[:, 2] - gt_boxes[:, 0]) * np.maximum(
        0.0, gt_boxes[:, 3] - gt_boxes[:, 1]
    )
    pred_area = np.maximum(0.0, pred_boxes[:, 2] - pred_boxes[:, 0]) * np.maximum(
        0.0, pred_boxes[:, 3] - pred_boxes[:, 1]
    )
    union_area = gt_area[:, None] + pred_area[None, :] - intersection_area

    return np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(intersection_area),
        where=union_area > 0,
    )


def match_frame(gt_entries, pred_entries, iou_threshold=IOU_THRESHOLD):
    """Greedily produce one-to-one matches, considering highest IoU first."""
    if not gt_entries or not pred_entries:
        return []

    iou_matrix = calculate_iou_matrix(
        [box for _, box in gt_entries], [box for _, box in pred_entries]
    )
    candidate_indices = np.argwhere(iou_matrix >= iou_threshold)
    if len(candidate_indices) == 0:
        return []

    candidate_scores = iou_matrix[candidate_indices[:, 0], candidate_indices[:, 1]]
    score_order = np.argsort(-candidate_scores, kind="stable")
    matched_gt_indices = set()
    matched_pred_indices = set()
    matches = []

    for candidate_position in score_order:
        gt_index, pred_index = candidate_indices[candidate_position]
        gt_index = int(gt_index)
        pred_index = int(pred_index)
        if gt_index in matched_gt_indices or pred_index in matched_pred_indices:
            continue

        matched_gt_indices.add(gt_index)
        matched_pred_indices.add(pred_index)
        matches.append((gt_entries[gt_index][0], pred_entries[pred_index][0]))

    return matches


def evaluate_tracking(ground_truth, predictions):
    """Return the predicted-ID count, average track length, and ID switches."""
    frames_by_prediction_id = defaultdict(set)
    for frame_id, frame_predictions in predictions.items():
        for person_id, _ in frame_predictions:
            frames_by_prediction_id[person_id].add(frame_id)

    number_of_ids = len(frames_by_prediction_id)
    average_track_length = (
        sum(len(frames) for frames in frames_by_prediction_id.values()) / number_of_ids
        if number_of_ids
        else 0.0
    )

    id_switches = 0
    last_gt_matches = {}
    for frame_id in sorted(ground_truth):
        matches = match_frame(
            ground_truth[frame_id], predictions.get(frame_id, []), IOU_THRESHOLD
        )
        for gt_id, prediction_id in matches:
            previous_match = last_gt_matches.get(gt_id)

            # An ID Switch occurs only when this GT target was also matched in the
            # immediately preceding frame and its associated prediction ID changed.
            if (
                previous_match is not None
                and previous_match[0] == frame_id - 1
                and previous_match[1] != prediction_id
            ):
                id_switches += 1

            last_gt_matches[gt_id] = (frame_id, prediction_id)

    return number_of_ids, average_track_length, id_switches


def main():
    args = parse_arguments()
    ground_truth = load_ground_truth(args.gt, args.max_frame)
    predictions = load_predictions(args.pred, args.max_frame)
    number_of_ids, average_track_length, id_switches = evaluate_tracking(
        ground_truth, predictions
    )

    print("========== Tracking Evaluation ==========")
    print()
    print("Prediction:")
    print(Path(args.pred).name)
    print()
    print("Number of IDs:")
    print(number_of_ids)
    print()
    print("Average Track Length:")
    print(f"{average_track_length:.2f} frames")
    print()
    print("ID Switch:")
    print(id_switches)


if __name__ == "__main__":
    main()
