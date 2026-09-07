"""Compare MediaPipe output with One Euro smoothing using the existing detector.

Coordinates remain normalized to the detector's person crop. Raw means the
output of the existing MediaPipe configuration (including its own smoothing).
Whole-sequence variance includes genuine motion and is not a pure noise metric.
"""

import argparse
import csv
import datetime
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LANDMARKS = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
}
CSV_FIELDS = [
    "frame_id", "person_id", "landmark_id", "raw_x", "raw_y",
    "filtered_x", "filtered_y", "visibility",
]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}
SUMMARY_FIELDS = [
    "video_name", "frame_count", "raw_y_std", "filtered_y_std",
    "noise_reduction", "raw_avg_displacement", "filtered_avg_displacement",
    "displacement_reduction", "raw_shoulder_p95_velocity", "filtered_shoulder_p95_velocity",
    "shoulder_velocity_retention",
]
# Experiment-only settings; each is applied before the first pose is processed.
PARAMETER_SETTINGS = [
    {"name": "A", "min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0},
    {"name": "B", "min_cutoff": 2.0, "beta": 0.007, "d_cutoff": 1.0},
    {"name": "C", "min_cutoff": 2.0, "beta": 0.02, "d_cutoff": 1.0},
    {"name": "D", "min_cutoff": 3.0, "beta": 0.02, "d_cutoff": 1.0},
]
COMPARISON_FIELDS = [
    "dataset_type", "video_name", "min_cutoff", "beta", "d_cutoff",
    "frame_count", "noise_reduction", "displacement_reduction",
    "shoulder_velocity_retention",
]


def summarize_shoulder_motion(shoulder_statistics):
    """Pool per-person shoulder-center velocities, never cross-person positions."""
    metrics = dict.fromkeys((
        "raw_shoulder_p95_velocity", "filtered_shoulder_p95_velocity",
        "shoulder_velocity_retention",
    ))
    raw_series = [v for stats in shoulder_statistics.values() for v in stats.raw_velocity_series]
    filtered_series = [
        v for stats in shoulder_statistics.values() for v in stats.filtered_velocity_series
    ]
    if raw_series:
        raw_p95 = float(np.percentile(raw_series, 95))
        filtered_p95 = float(np.percentile(filtered_series, 95))
        metrics["raw_shoulder_p95_velocity"] = raw_p95
        metrics["filtered_shoulder_p95_velocity"] = filtered_p95
        if raw_p95 > 0:
            metrics["shoulder_velocity_retention"] = filtered_p95 / raw_p95 * 100.0
    return metrics


def summarize_video(video_name, frame_count, statistics, shoulder_statistics=None):
    """Aggregate within-track variance and valid consecutive-frame displacements.

    Variances are weighted by sample count, without mixing track mean positions.
    Percentages are stored on a 0..100 scale (negative/>100 values are possible).
    Velocity is Euclidean displacement between consecutive frames, without /dt.
    P95 retention uses only shoulder-center velocities, computed per person
    before pooling. Other metrics continue to use the five selected landmarks.
    P95 uses NumPy's default linear percentile interpolation.
    Displacement reduction = (1 - filtered_avg / raw_avg) * 100.
    Undefined metrics remain blank in CSV, rather than implying perfect smoothing.
    """
    row = dict.fromkeys(SUMMARY_FIELDS)
    row.update(video_name=video_name, frame_count=frame_count)
    tracks = list(statistics.values())
    samples = sum(track.count for track in tracks)
    pairs = sum(track.pair_count for track in tracks)
    if samples:
        raw_std, filtered_std = [
            math.sqrt(max(0.0, sum(track.y_m2[i] for track in tracks) / samples))
            for i in (0, 1)
        ]
        row.update(raw_y_std=raw_std, filtered_y_std=filtered_std)
        if raw_std > 0:
            row["noise_reduction"] = 100.0 * (1.0 - filtered_std / raw_std)
    if pairs:
        row["raw_avg_displacement"], row["filtered_avg_displacement"] = [
            sum(track.displacement_sum[i] for track in tracks) / pairs
            for i in (0, 1)
        ]
        if row["raw_avg_displacement"] > 0:
            row["displacement_reduction"] = (
                1.0 - row["filtered_avg_displacement"] / row["raw_avg_displacement"]
            ) * 100.0
    row.update(summarize_shoulder_motion(shoulder_statistics or {}))
    return row


class ShoulderMotionStatistics:
    """Store raw/filtered center velocities for one person's shoulders (11, 12)."""

    def __init__(self):
        self.previous = None
        self.raw_velocity_series = []
        self.filtered_velocity_series = []

    def update(self, frame_id, raw_landmarks, filtered_landmarks):
        if len(raw_landmarks) <= 12 or len(filtered_landmarks) <= 12:
            self.previous = None
            return
        # Average coordinates FIRST; averaging the two shoulder speeds would
        # give a different result, especially for opposite shoulder movements.
        centers = tuple(
            tuple((points[11][axis] + points[12][axis]) / 2.0 for axis in (0, 1))
            for points in (raw_landmarks, filtered_landmarks)
        )
        if self.previous is not None and self.previous[0] == frame_id - 1:
            for center, previous, series in zip(
                centers, self.previous[1],
                (self.raw_velocity_series, self.filtered_velocity_series),
            ):
                series.append(math.hypot(center[0] - previous[0], center[1] - previous[1]))
        self.previous = (frame_id, centers)


class LandmarkStatistics:
    """Online per-landmark variance and displacement statistics."""

    def __init__(self):
        self.count = 0
        self.y_mean = [0.0, 0.0]
        self.y_m2 = [0.0, 0.0]
        self.pair_count = 0
        self.change_count = 0
        self.displacement_sum = [0.0, 0.0]
        self.max_displacement_change = [0.0, 0.0]
        self.previous = None
        self.previous_displacement = None

    def update(self, frame_id, raw, filtered):
        self.count += 1
        points = (raw, filtered)
        for index, point in enumerate(points):
            delta = point[1] - self.y_mean[index]
            self.y_mean[index] += delta / self.count
            self.y_m2[index] += delta * (point[1] - self.y_mean[index])

        # Never connect different people or bridge frames with missing poses.
        if self.previous is not None and self.previous[0] == frame_id - 1:
            displacement = [
                math.hypot(point[0] - previous[0], point[1] - previous[1])
                for point, previous in zip(points, self.previous[1])
            ]
            self.pair_count += 1
            for index, value in enumerate(displacement):
                self.displacement_sum[index] += value
                if self.previous_displacement is not None:
                    self.max_displacement_change[index] = max(
                        self.max_displacement_change[index],
                        abs(value - self.previous_displacement[index]),
                    )
            if self.previous_displacement is not None:
                self.change_count += 1
            self.previous_displacement = displacement
        else:
            self.previous_displacement = None
        self.previous = (frame_id, points)

    def y_std(self):
        """Population standard deviation over observed frames."""
        return [math.sqrt(max(0.0, value / self.count)) for value in self.y_m2]


class PoseRecordingMixin:
    """Intercept the existing smoothing hook; run MediaPipe only once per pose."""

    def smooth_pose_landmarks(self, landmarks, person_id, timestamp):
        # Use source-video time so slow inference does not change filter strength.
        # The production detector continues to use its original monotonic clock.
        filtered = super().smooth_pose_landmarks(
            landmarks, person_id, self.video_timestamp
        )
        if person_id not in self.shoulder_statistics:
            self.shoulder_statistics[person_id] = ShoulderMotionStatistics()
        self.shoulder_statistics[person_id].update(self.frame_id, landmarks, filtered)
        for landmark_id, (raw, smoothed) in enumerate(zip(landmarks, filtered)):
            if self.csv_writer is not None:
                self.csv_writer.writerow([
                    self.frame_id, person_id, landmark_id,
                    raw[0], raw[1], smoothed[0], smoothed[1], raw[3],
                ])
            if landmark_id in LANDMARKS:
                key = (person_id, landmark_id)
                if key not in self.statistics:
                    self.statistics[key] = LandmarkStatistics()
                self.statistics[key].update(self.frame_id, raw[:2], smoothed[:2])
        return filtered


def print_results(statistics, log_path, shoulder_statistics=None):
    print("========== Pose Smoothing Evaluation ==========")
    print(f"CSV: {log_path}")
    print("Coordinates: normalized person-crop coordinates; visibility is unmodified.")
    print("Raw: current MediaPipe output, before the added One Euro filter.")
    print("Noise Reduction is Y-std reduction, including genuine motion; not a pure noise score.")
    print("Compare stationary and fast-motion clips separately; inspect CSV trajectories for lag.")
    if not statistics:
        print("No pose landmarks detected; CSV contains only the header.")
        return

    for (person_id, landmark_id), stats in sorted(statistics.items()):
        raw_std, filtered_std = stats.y_std()
        reduction = (
            f"{100.0 * (1.0 - filtered_std / raw_std):.2f}%"
            if raw_std > 0 else "N/A (zero raw variance)"
        )
        print(f"\nPerson ID: {person_id}")
        print(f"Landmark:\n{LANDMARKS[landmark_id]}")
        print(f"Samples: {stats.count}; consecutive frame pairs: {stats.pair_count}")
        print(f"Raw Y Std:\n{raw_std:.6f}")
        print(f"Filtered Y Std:\n{filtered_std:.6f}")
        print(f"Noise Reduction:\n{reduction}")
        print("Average Displacement:")
        for index, label in enumerate(("raw", "filtered")):
            value = (
                f"{stats.displacement_sum[index] / stats.pair_count:.6f}"
                if stats.pair_count else "N/A (no consecutive frames)"
            )
            print(f"{label} {value}")
        # Reuse the summary formulas for each individual person/landmark report.
        metrics = summarize_video("", 0, {(person_id, landmark_id): stats})
        value = metrics["displacement_reduction"]
        formatted = f"{value:.2f}%" if value is not None else "N/A"
        print(f"Average Displacement Reduction:\n{formatted}")
        print("Max Displacement Change (max |d_t - d_(t-1)|):")
        for index, label in enumerate(("raw", "filtered")):
            value = (
                f"{stats.max_displacement_change[index]:.6f}"
                if stats.change_count else "N/A (needs 3 consecutive frames)"
            )
            print(f"{label} {value}")

    for person_id, stats in sorted((shoulder_statistics or {}).items()):
        print(f"\nPerson ID: {person_id}\nMotion point: shoulder center (11, 12)")
        metrics = summarize_shoulder_motion({person_id: stats})
        for field, label in (
            ("raw_shoulder_p95_velocity", "Raw Shoulder P95 Velocity"),
            ("filtered_shoulder_p95_velocity", "Filtered Shoulder P95 Velocity"),
            ("shoulder_velocity_retention", "Shoulder Velocity Retention"),
        ):
            value = metrics[field]
            formatted = "N/A" if value is None else (
                f"{value:.2f}%" if field == "shoulder_velocity_retention" else f"{value:.6f}"
            )
            print(f"{label}:\n{formatted}")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--source", help="Path to an input video")
    sources.add_argument("--source-dir", type=Path, help="Directory of MP4, AVI or MOV videos")
    sources.add_argument(
        "--normal-dir", type=Path,
        help="Normal-action directory for parameter comparison; requires --fall-dir",
    )
    parser.add_argument("--fall-dir", type=Path, help="Fall-video directory for parameter comparison")
    parser.add_argument("--model", default="yolov12n1.pt", help="Existing YOLO model path")
    parser.add_argument("--conf", type=float, default=0.5, help="Pose/analysis threshold")
    parser.add_argument("--detector-conf", type=float, default=0.3, help="YOLO threshold")
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override video FPS for filter timestamps (required if metadata is invalid)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "logs",
        help="Directory for the experiment CSV",
    )
    args = parser.parse_args()
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    if args.source is not None and not Path(args.source).is_file():
        parser.error(f"Video file does not exist: {args.source}")
    if args.source_dir is not None and not args.source_dir.is_dir():
        parser.error(f"Video directory does not exist: {args.source_dir}")
    if (args.normal_dir is None) != (args.fall_dir is None):
        parser.error("Parameter comparison requires both --normal-dir and --fall-dir")
    for directory in (args.normal_dir, args.fall_dir):
        if directory is not None and not directory.is_dir():
            parser.error(f"Video directory does not exist: {directory}")
    return args


def process_video(source, log_path, args, detector_class, cv2, filter_params=None):
    """Create a fresh detector; log_path=None collects summary metrics only."""
    cap = cv2.VideoCapture(str(source))
    detector = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        fps = args.fps if args.fps is not None else cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid video FPS; provide --fps with the source frame rate")

        # Constructor initializes tracker, pose_filters, pose_histories and Pose.
        detector = detector_class(
            model_path=args.model,
            confidence=args.conf,
            detector_confidence=args.detector_conf,
        )
        if filter_params is not None:
            # Filters are created lazily by the existing smoothing hook. Override
            # this instance's settings before the first frame, not global defaults.
            detector.pose_filter_params = dict(filter_params)
        detector.statistics = {}
        detector.shoulder_statistics = {}
        detector.frame_id = 0
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Source: {source}; filter timebase: {fps:g} FPS")
        print(f"One Euro parameters: {detector.pose_filter_params}")
        if log_path is not None:
            print(f"CSV: {log_path}")
        csv_context = (
            log_path.open("w", newline="", encoding="utf-8")
            if log_path is not None else nullcontext(None)
        )
        with csv_context as csv_file:
            detector.csv_writer = csv.writer(csv_file) if csv_file is not None else None
            if detector.csv_writer is not None:
                detector.csv_writer.writerow(CSV_FIELDS)
            while True:
                received, frame = cap.read()
                if not received:
                    break
                detector.frame_id += 1
                detector.video_timestamp = (detector.frame_id - 1) / fps
                detector.process_frame(frame)
                if detector.frame_id % 100 == 0:
                    print(f"Processed {detector.frame_id} frames", flush=True)
        if log_path is not None:
            print_results(detector.statistics, log_path, detector.shoulder_statistics)
        return summarize_video(
            Path(source).name, detector.frame_id, detector.statistics, detector.shoulder_statistics
        )
    finally:
        cap.release()
        if detector is not None:
            detector.close()


def find_videos(source_dir):
    """Search the immediate directory; match extensions case-insensitively."""
    return sorted(
        (path for path in source_dir.iterdir()
         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def batch_csv_paths(videos, output_dir):
    """Keep stems when unique; distinguish e.g. walking.mp4 and walking.avi."""
    used_names = set()
    paths = []
    for video in videos:
        name = f"{video.stem}.csv"
        index = 1
        while name.lower() in used_names:
            name = f"{video.name}_{index}.csv"
            index += 1
        used_names.add(name.lower())
        paths.append(output_dir / name)
    return paths


def run_parameter_comparison(args, datasets, detector_class, cv2):
    """Run settings A-D over both datasets, writing exactly one new CSV."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary_path = args.output_dir / f"one_euro_parameter_comparison_{timestamp}.csv"
    total = len(PARAMETER_SETTINGS) * sum(len(videos) for _, videos in datasets)
    completed = 0
    failures = 0
    # Exclusive creation protects existing results even in a filename collision.
    with summary_path.open("x", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        print(f"Parameter comparison CSV: {summary_path}", flush=True)
        for setting in PARAMETER_SETTINGS:
            params = {key: setting[key] for key in ("min_cutoff", "beta", "d_cutoff")}
            for dataset_type, videos in datasets:
                for source in videos:
                    print(f"[{completed + 1}/{total}] Setting {setting['name']}, {dataset_type}: {source.name}", flush=True)
                    row = dict.fromkeys(COMPARISON_FIELDS)
                    row.update(dataset_type=dataset_type, video_name=source.name, **params)
                    try:
                        metrics = process_video(
                            source, None, args, detector_class, cv2, filter_params=params
                        )
                        for field in (
                            "frame_count", "noise_reduction", "displacement_reduction",
                            "shoulder_velocity_retention",
                        ):
                            row[field] = metrics[field]
                    except (OSError, RuntimeError, ValueError) as error:
                        failures += 1
                        print(f"Failed: {setting['name']}, {dataset_type}, {source}: {error}", file=sys.stderr)
                    writer.writerow(row)
                    summary_file.flush()
                    completed += 1
    print("========== One Euro Parameter Comparison ==========")
    print(f"Video/parameter runs: {completed}; failed: {failures}")
    print("Percentages are numeric; undefined metrics and failed-run metrics are blank.")
    print(f"Summary CSV: {summary_path}")
    return summary_path


def main():
    args = parse_arguments()
    videos = find_videos(args.source_dir) if args.source_dir is not None else None
    if videos == []:
        raise SystemExit(f"No MP4, AVI or MOV videos found in: {args.source_dir}")
    datasets = None
    if args.normal_dir is not None:
        datasets = [("normal", find_videos(args.normal_dir)), ("fall", find_videos(args.fall_dir))]
        for dataset_type, dataset_videos in datasets:
            if not dataset_videos:
                raise SystemExit(f"No MP4, AVI or MOV videos found in the {dataset_type} directory")

    # Delay model imports so --help and statistics can run without model loading.
    import cv2
    from models.fall_detector import FallDetector

    class RecordingFallDetector(PoseRecordingMixin, FallDetector):
        pass

    if datasets is not None:
        run_parameter_comparison(args, datasets, RecordingFallDetector, cv2)
        return

    if videos is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_path = args.output_dir / f"pose_smoothing_{timestamp}.csv"
        process_video(args.source, log_path, args, RecordingFallDetector, cv2)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "pose_smoothing_summary.csv"
    log_paths = batch_csv_paths(videos, args.output_dir / "pose_smoothing")
    rows = []
    failures = 0
    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for source, log_path in zip(videos, log_paths):
            try:
                row = process_video(source, log_path, args, RecordingFallDetector, cv2)
            except (OSError, RuntimeError, ValueError) as error:
                failures += 1
                print(f"Failed: {source}: {error}", file=sys.stderr)
                print(f"CSV may be absent or partial: {log_path}", file=sys.stderr)
                row = dict.fromkeys(SUMMARY_FIELDS)
                row["video_name"] = source.name
            writer.writerow(row)
            summary_file.flush()
            rows.append(row)

    print("========== Pose Smoothing Batch Evaluation ==========")
    print(f"Videos:\n{len(videos)}")
    if failures:
        print(f"Failed videos: {failures} (blank summary metrics)")
    for field, label in (
        ("noise_reduction", "Average Noise Reduction"),
        ("shoulder_velocity_retention", "Average Shoulder Velocity Retention"),
        ("displacement_reduction", "Average Displacement Reduction"),
    ):
        values = [row[field] for row in rows if row[field] is not None]
        average = f"{sum(values) / len(values):.2f}%" if values else "N/A"
        print(f"{label}:\n{average}")
        print(f"Valid videos: {len(values)}/{len(videos)}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
