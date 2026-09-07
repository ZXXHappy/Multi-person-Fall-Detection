"""Measure raw MediaPipe visibility through the project's existing Pose pipeline.

No additional smoothing is applied. The existing MediaPipe configuration is
preserved. Counts represent observed person-frame samples, not unique frames;
missing detections/poses are not assigned artificial zero visibility.
"""

import argparse
import csv
import datetime
import math
import statistics
import sys
from itertools import chain
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KEYPOINTS = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
}
VIDEO_EXTENSIONS = {".mp4", ".avi"}
THRESHOLDS = [
    ("ratio_lt_0.1", "<0.1", 0.1, False),
    ("ratio_lt_0.2", "<0.2", 0.2, False),
    ("ratio_lt_0.3", "<0.3", 0.3, False),
    ("ratio_lt_0.5", "<0.5", 0.5, False),
    ("ratio_ge_0.5", ">=0.5", 0.5, True),
    ("ratio_ge_0.7", ">=0.7", 0.7, True),
    ("ratio_ge_0.9", ">=0.9", 0.9, True),
]
SUMMARY_FIELDS = ["keypoint", "frame_count", "mean", "median", "std"] + [
    field for field, _, _, _ in THRESHOLDS
]
SAMPLE_FIELDS = ["video_name", "keypoint", "visibility"]


class VisibilityRecordingMixin:
    """Record at the detector's smoothing hook and return untouched landmarks."""

    def smooth_pose_landmarks(self, landmarks, person_id, timestamp):
        # Deliberately do not call super(): that would invoke One Euro smoothing.
        for landmark_id, name in KEYPOINTS.items():
            if landmark_id >= len(landmarks):
                continue
            visibility = float(landmarks[landmark_id][3])
            if not math.isfinite(visibility):
                self.invalid_visibility_count += 1
                continue
            self.visibility_samples[name].append(visibility)
            self.sample_writer.writerow([self.video_name, name, visibility])
        return landmarks


def summarize_visibility(keypoint, values):
    """Population std and overlapping threshold proportions, stored as 0..1."""
    row = dict.fromkeys(SUMMARY_FIELDS)
    row.update(keypoint=keypoint, frame_count=len(values))
    if not values:
        return row
    row.update(
        mean=statistics.fmean(values),
        median=statistics.median(values),
        std=statistics.pstdev(values),
    )
    for field, _, threshold, greater_equal in THRESHOLDS:
        count = sum(
            value >= threshold if greater_equal else value < threshold
            for value in values
        )
        row[field] = count / len(values)
    return row


def print_report(rows):
    print("========== Visibility Analysis ==========")
    print("Frames counts observed person-frame samples; missing poses are excluded.")
    print("Overall pools all five keypoints. CSV ratios use 0..1; terminal shows percent.")
    for row in rows:
        title = "Overall visibility distribution" if row["keypoint"] == "overall" else row["keypoint"]
        print(f"\nKeypoint: {title}")
        print(f"Frames:\n{row['frame_count']}")
        for field, label in (("mean", "Mean"), ("median", "Median"), ("std", "Std")):
            value = row[field]
            formatted = f"{value:.6f}" if value is not None else "N/A"
            print(f"{label}:\n{formatted}")
        print("Visibility Distribution:")
        for field, label, _, _ in THRESHOLDS:
            value = row[field]
            formatted = f"{value * 100:.2f}%" if value is not None else "N/A"
            print(f"{label}:\n{formatted}")


def find_videos(source_dir):
    """List videos immediately inside the directory, case-insensitive extensions."""
    return sorted(
        (path for path in source_dir.iterdir()
         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", "--source-dir", type=Path, required=True,
                        help="Directory of MP4 and AVI videos")
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
    """Run the existing detector, resetting all per-person state for each video."""
    cap = cv2.VideoCapture(str(source))
    detector = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        detector = detector_class(
            model_path=args.model, confidence=args.conf,
            detector_confidence=args.detector_conf,
        )
        detector.visibility_samples = samples
        detector.sample_writer = sample_writer
        detector.video_name = source.name
        detector.invalid_visibility_count = 0
        frame_count = 0
        while True:
            received, frame = cap.read()
            if not received:
                break
            detector.process_frame(frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"{source.name}: processed {frame_count} frames", flush=True)
        print(f"{source.name}: {frame_count} frames; invalid visibility skipped: "
              f"{detector.invalid_visibility_count}")
    finally:
        cap.release()
        if detector is not None:
            detector.close()


def main():
    args = parse_arguments()
    videos = find_videos(args.source_dir)
    if not videos:
        raise SystemExit(f"No MP4 or AVI videos found in: {args.source_dir}")

    # Reuse the production inference pipeline; no new Pose model or configuration.
    import cv2
    from models.fall_detector import FallDetector

    class VisibilityFallDetector(VisibilityRecordingMixin, FallDetector):
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_path = args.output_dir / f"visibility_distribution_{timestamp}.csv"
    sample_path = args.output_dir / f"visibility_samples_{timestamp}.csv"
    samples = {name: [] for name in KEYPOINTS.values()}
    failures = []
    print(f"Videos: {len(videos)}\nRaw visibility samples: {sample_path}", flush=True)
    with sample_path.open("x", newline="", encoding="utf-8") as sample_file:
        writer = csv.writer(sample_file)
        writer.writerow(SAMPLE_FIELDS)
        for source in videos:
            try:
                process_video(source, args, VisibilityFallDetector, cv2, samples, writer)
            except (OSError, RuntimeError, ValueError) as error:
                failures.append(source.name)
                print(f"Failed: {source}: {error}. Already recorded samples are retained.",
                      file=sys.stderr)
            sample_file.flush()

    rows = [summarize_visibility(name, values) for name, values in samples.items()]
    rows.append(summarize_visibility("overall", list(chain.from_iterable(samples.values()))))
    with summary_path.open("x", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_report(rows)
    if failures:
        print(f"Failed/partial videos: {', '.join(failures)}")
    print(f"Summary CSV: {summary_path}\nSample CSV: {sample_path}")


if __name__ == "__main__":
    main()
