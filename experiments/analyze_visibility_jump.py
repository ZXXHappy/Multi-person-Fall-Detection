"""Compare raw MediaPipe displacement across current-frame visibility groups.

Displacements use normalized person-crop coordinates from the existing detector.
They may include real motion and crop changes as well as landmark jitter; a large
displacement alone does not establish an erroneous pose estimate.
"""

import argparse
import csv
import datetime
import math
import statistics
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_visibility_distribution import KEYPOINTS, find_videos


GROUPS = {
    "A": "visibility >= 0.5",
    "B": "0.3 <= visibility < 0.5",
    "C": "visibility < 0.3",
}
CSV_FIELDS = ["video_name", "keypoint", "visibility_group", "frame", "visibility", "displacement"]


def visibility_group(visibility):
    if visibility >= 0.5:
        return "A"
    if visibility >= 0.3:
        return "B"
    return "C"


def new_displacement_samples():
    return {name: {group: [] for group in GROUPS} for name in KEYPOINTS.values()}


class VisibilityJumpMixin:
    """Measure raw landmark motion without calling the One Euro smoothing hook."""

    def smooth_pose_landmarks(self, landmarks, person_id, timestamp):
        for landmark_id, name in KEYPOINTS.items():
            key = (person_id, landmark_id)
            if landmark_id >= len(landmarks):
                self.previous_landmarks.pop(key, None)
                continue
            x, y, _, visibility = landmarks[landmark_id]
            if not all(math.isfinite(value) for value in (x, y, visibility)):
                self.previous_landmarks.pop(key, None)
                self.invalid_sample_count += 1
                continue

            previous = self.previous_landmarks.get(key)
            # First observations and gaps yield no velocity sample. Compare
            # coordinates only within the same person, landmark and video.
            if previous is not None and previous[0] == self.frame_id - 1:
                displacement = math.hypot(x - previous[1], y - previous[2])
                group = visibility_group(visibility)  # Use the CURRENT frame.
                self.displacement_samples[name][group].append(displacement)
                self.sample_writer.writerow([
                    self.video_name, name, group, self.frame_id, visibility, displacement,
                ])
            self.previous_landmarks[key] = (self.frame_id, x, y)

        # Do not call super(): return the exact raw list without extra filtering.
        return landmarks


def summarize_displacements(values):
    if not values:
        return {"sample_count": 0, "mean": None, "median": None, "max": None, "p95": None}
    return {
        "sample_count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
        "p95": float(np.percentile(values, 95)),
    }


def print_report(samples):
    print("========== Visibility Jump Analysis ==========")
    print("Groups use current-frame visibility; counts are person/landmark frame pairs.")
    print("Missing poses and first observations do not contribute displacements.")
    print("Units: normalized person-crop coordinates per frame; no One Euro filtering.")
    overall = {
        group: [value for keypoint in samples.values() for value in keypoint[group]]
        for group in GROUPS
    }
    for keypoint, grouped in [*samples.items(), ("overall", overall)]:
        print(f"\nKeypoint: {keypoint}")
        for group, description in GROUPS.items():
            metrics = summarize_displacements(grouped[group])
            print(f"Group {group}: {description}")
            print(f"Sample Count: {metrics['sample_count']}")
            for field, label in (
                ("mean", "Mean Displacement"),
                ("median", "Median Displacement"),
                ("max", "Max Displacement"),
                ("p95", "95 Percentile Displacement"),
            ):
                value = metrics[field]
                formatted = f"{value:.6f}" if value is not None else "N/A"
                print(f"{label}: {formatted}")
    print("\nCompare B/C with A, checking sample counts as well as median and P95.")
    print("Larger low-visibility displacements suggest an association, not proof of errors.")
    print("Real motion, bounding-box crop changes and group sample imbalance also affect results.")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", "--source-dir", type=Path, required=True,
                        help="Directory containing MP4 and AVI videos")
    parser.add_argument("--model", default="yolov12n1.pt", help="Existing YOLO model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Pose/analysis threshold")
    parser.add_argument("--detector-conf", type=float, default=0.3, help="YOLO threshold")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "logs")
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        parser.error(f"Video directory does not exist: {args.source_dir}")
    return args


def process_video(source, args, detector_class, cv2, samples, sample_writer):
    cap = cv2.VideoCapture(str(source))
    detector = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        detector = detector_class(
            model_path=args.model, confidence=args.conf,
            detector_confidence=args.detector_conf,
        )
        detector.previous_landmarks = {}
        detector.displacement_samples = samples
        detector.sample_writer = sample_writer
        detector.video_name = source.name
        detector.frame_id = 0
        detector.invalid_sample_count = 0
        while True:
            received, frame = cap.read()
            if not received:
                break
            detector.frame_id += 1
            detector.process_frame(frame)
            # Retain only this frame's positions: absent IDs cannot bridge gaps.
            detector.previous_landmarks = {
                key: value for key, value in detector.previous_landmarks.items()
                if value[0] == detector.frame_id
            }
            if detector.frame_id % 100 == 0:
                print(f"{source.name}: {detector.frame_id} frames", flush=True)
        print(f"{source.name}: {detector.frame_id} frames; invalid samples skipped: "
              f"{detector.invalid_sample_count}")
    finally:
        cap.release()
        if detector is not None:
            detector.close()


def main():
    args = parse_arguments()
    videos = find_videos(args.source_dir)
    if not videos:
        raise SystemExit(f"No MP4 or AVI videos found in: {args.source_dir}")

    import cv2
    from models.fall_detector import FallDetector

    class VisibilityJumpDetector(VisibilityJumpMixin, FallDetector):
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = args.output_dir / f"visibility_jump_analysis_{timestamp}.csv"
    samples = new_displacement_samples()
    failures = []
    print(f"Videos: {len(videos)}\nCSV: {log_path}", flush=True)
    with log_path.open("x", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(CSV_FIELDS)
        for source in videos:
            try:
                process_video(source, args, VisibilityJumpDetector, cv2, samples, writer)
            except (OSError, RuntimeError, ValueError) as error:
                failures.append(source.name)
                print(f"Failed: {source}: {error}. Already recorded samples are retained.",
                      file=sys.stderr)
            log_file.flush()
    print_report(samples)
    if failures:
        print(f"Failed/partial videos: {', '.join(failures)}")
    print(f"CSV: {log_path}")


if __name__ == "__main__":
    main()
