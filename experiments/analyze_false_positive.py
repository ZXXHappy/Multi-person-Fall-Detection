"""Record per-person context around alarms in videos labelled as normal.

Observes the production One Euro auxiliary branch and Transformer alarms,
including its native filter clock. Coordinates/velocities are the post-filter
values used by the detector. Visibility remains the current MediaPipe score.
Hip/shoulder heights mean normalized crop Y coordinates (larger = lower).
Bounding-box dimensions are detector-box pixels; pose_aspect_ratio is the actual
pose ratio retained for auxiliary analysis. No new alarm criterion is introduced here.
"""

import argparse
import csv
import datetime
import math
import statistics
import sys
from collections import deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.analyze_visibility_distribution import find_videos


CSV_FIELDS = [
    "video_name", "alarm_frame", "frame", "hip_velocity", "shoulder_velocity",
    "body_angle", "hip_height", "bbox_height", "bbox_width", "bbox_ratio",
    "left_hip_visibility", "right_hip_visibility",
    "left_shoulder_visibility", "right_shoulder_visibility",
    # Extra fields preserve the requested time and distinguish simultaneous people.
    "person_id", "alarm_time", "left_hip_velocity", "right_hip_velocity",
    "mean_velocity", "max_velocity", "shoulder_height", "pose_aspect_ratio",
]
VISIBILITY_FIELDS = (
    "left_hip_visibility", "right_hip_visibility",
    "left_shoulder_visibility", "right_shoulder_visibility",
)


def finite_values(values):
    return [value for value in values if value is not None and math.isfinite(value)]


def mean_max(values):
    valid = finite_values(values)
    return (statistics.fmean(valid), max(valid)) if valid else (None, None)


class FeatureRecordingMixin:
    """Observe the existing analyze_pose result without changing its inputs/output."""

    def analyze_pose(self, frame, person_box, person_id):
        landmarks, features = super().analyze_pose(frame, person_box, person_id)
        if not landmarks or features is None or len(landmarks) <= 24:
            self.previous_positions.pop(person_id, None)
            return landmarks, features

        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]
        shoulder = tuple((landmarks[11][i] + landmarks[12][i]) / 2.0 for i in (0, 1))
        points = (left_hip, right_hip, shoulder)
        previous = self.previous_positions.get(person_id)
        velocities = [None, None, None]
        if previous is not None and previous[0] == self.analysis_frame - 1:
            velocities = [
                math.hypot(point[0] - old[0], point[1] - old[1])
                for point, old in zip(points, previous[1])
            ]
        self.previous_positions[person_id] = (self.analysis_frame, points)
        left_velocity, right_velocity, shoulder_velocity = velocities
        hip_velocity = (
            (left_velocity + right_velocity) / 2.0
            if left_velocity is not None and right_velocity is not None else None
        )
        average, maximum = mean_max(velocities)
        x1, y1, x2, y2 = person_box
        width, height = float(x2 - x1), float(y2 - y1)
        self.frame_observations[person_id] = {
            "hip_velocity": hip_velocity,
            "left_hip_velocity": left_velocity,
            "right_hip_velocity": right_velocity,
            "shoulder_velocity": shoulder_velocity,
            "mean_velocity": average,
            "max_velocity": maximum,
            "body_angle": features["angle"],
            "hip_height": features["mid_hip_y"],
            "shoulder_height": features["mid_shoulder_y"],
            "bbox_height": height,
            "bbox_width": width,
            "bbox_ratio": height / width if width > 0 else None,
            "pose_aspect_ratio": features["aspect_ratio"],
            "left_hip_visibility": landmarks[23][3],
            "right_hip_visibility": landmarks[24][3],
            "left_shoulder_visibility": landmarks[11][3],
            "right_shoulder_visibility": landmarks[12][3],
        }
        return landmarks, features


class AlarmWindowRecorder:
    """Buffer frame snapshots; each person alarm owns an independent window."""

    def __init__(self, video_name, fps, writer, before=30, after=10):
        self.video_name = video_name
        self.fps = fps
        self.writer = writer
        self.before = before
        self.after = after
        self.history = deque(maxlen=before + 1)
        self.active = []
        self.previous_alarm_ids = set()
        self.alarm_samples = []

    def record_frame(self, frame_id, observations, alarm_ids, inferred_event_ids):
        self.history.append((frame_id, observations))
        remaining = []
        for event in self.active:
            event["rows"].append((frame_id, observations.get(event["person_id"], {})))
            if frame_id >= event["alarm_frame"] + self.after:
                self.write_event(event)
            else:
                remaining.append(event)
        self.active = remaining

        # A continuing alarm for the same ID is one event. A new ID can alarm
        # while another ID is active; never merge different people's events.
        current_alarm_ids = set(alarm_ids)
        for person_id in sorted(set(inferred_event_ids) - self.previous_alarm_ids):
            metadata = {
                "video_name": self.video_name,
                "person_id": person_id,
                "alarm_frame": frame_id,
                "alarm_time": (frame_id - 1) / self.fps,
            }
            self.alarm_samples.append({**metadata, **observations.get(person_id, {})})
            event = {
                **metadata,
                "rows": [(index, people.get(person_id, {})) for index, people in self.history],
            }
            if self.after == 0:
                self.write_event(event)
            else:
                self.active.append(event)
        self.previous_alarm_ids = current_alarm_ids

    def write_event(self, event):
        for frame_id, sample in event["rows"]:
            row = dict.fromkeys(CSV_FIELDS)
            row.update({key: event[key] for key in ("video_name", "person_id", "alarm_frame", "alarm_time")})
            row.update(frame=frame_id, **sample)
            self.writer.writerow(row)
        mean, maximum = mean_max(sample.get("hip_velocity") for _, sample in event["rows"])
        print(f"Alarm: {self.video_name}, person={event['person_id']}, "
              f"frame={event['alarm_frame']}, time={event['alarm_time']:.3f}s; "
              f"window={event['rows'][0][0]}..{event['rows'][-1][0]}; "
              f"window hip velocity mean={mean}, max={maximum}")

    def finish(self):
        # End-of-video clips have shorter post-alarm windows; do not fabricate frames.
        for event in self.active:
            self.write_event(event)
        self.active.clear()


def analyze_video(source, args, detector_class, detector_module, cv2, writer):
    cap = cv2.VideoCapture(str(source))
    detector = None
    recorder = None
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {source}")
        fps = args.fps if args.fps is not None else cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid FPS for {source}; supply the video frame rate with --fps")
        detector = detector_class(
            model_path=args.model, confidence=args.conf,
            detector_confidence=args.detector_conf,
        )
        detector.previous_positions = {}
        detector.analysis_frame = 0
        recorder = AlarmWindowRecorder(source.name, fps, writer, args.before, args.after)
        print(f"Analyzing {source.name}: Transformer alarms, One Euro auxiliary features; FPS={fps:g}", flush=True)
        while True:
            received, frame = cap.read()
            if not received:
                break
            detector.analysis_frame += 1
            detector.frame_observations = {}
            _, alarm, fall_data = detector.process_frame(frame)
            recorder.record_frame(detector.analysis_frame, detector.frame_observations,
                                  fall_data["fallen_ids"], fall_data["fall_event_ids"])
            detector.previous_positions = {
                person_id: value for person_id, value in detector.previous_positions.items()
                if value[0] == detector.analysis_frame
            }
            if detector.analysis_frame % 100 == 0:
                print(f"{source.name}: {detector.analysis_frame} frames", flush=True)
        if detector.analysis_frame == 0:
            raise RuntimeError(f"No readable frames: {source}")
        return recorder.alarm_samples
    finally:
        try:
            if recorder is not None:
                recorder.finish()
        finally:
            cap.release()
            if detector is not None:
                detector.close()


def print_summary(video_count, false_alarm_videos, alarm_samples):
    print("========== False Positive Analysis ==========")
    print(f"Total normal videos:\n{video_count}")
    print(f"False alarm videos:\n{false_alarm_videos}")
    print(f"Total false alarms:\n{len(alarm_samples)}")
    print("Statistics below use each event's alarm frame once, not overlapping window rows.")
    print("Body angle statistics use absolute degrees; velocity is normalized displacement/frame.")
    for name, values in (
        ("Hip velocity", [sample.get("hip_velocity") for sample in alarm_samples]),
        ("Body angle", [abs(sample["body_angle"]) for sample in alarm_samples if sample.get("body_angle") is not None]),
    ):
        mean, maximum = mean_max(values)
        print(f"{name}:")
        for label, value in (("mean", mean), ("max", maximum)):
            formatted = f"{value:.6f}" if value is not None else "N/A"
            print(f"{label}:\n{formatted}")
    visibilities = finite_values(sample.get(field) for sample in alarm_samples for field in VISIBILITY_FIELDS)
    visibility_mean = f"{statistics.fmean(visibilities):.6f}" if visibilities else "N/A"
    print(f"Visibility:\nmean:\n{visibility_mean}")
    print("Compare visibility dips, motion peaks and pose_aspect_ratio around each alarm.")
    print("These associations do not by themselves establish the cause of a false alarm.")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", "--source-dir", type=Path, required=True)
    parser.add_argument("--before", type=int, default=30, help="Frames before each alarm")
    parser.add_argument("--after", type=int, default=10, help="Frames after each alarm")
    parser.add_argument("--model", default="yolov12n1.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--detector-conf", type=float, default=0.3)
    parser.add_argument("--fps", type=float, default=None, help="Override FPS for alarm times only")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "logs")
    args = parser.parse_args()
    if not args.source_dir.is_dir():
        parser.error(f"Video directory does not exist: {args.source_dir}")
    if args.before < 0 or args.after < 0:
        parser.error("--before and --after must be nonnegative")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be finite and positive")
    return args


def main():
    args = parse_arguments()
    videos = find_videos(args.source_dir)
    if not videos:
        raise SystemExit(f"No MP4 or AVI videos in: {args.source_dir}")
    import cv2
    import models.fall_detector as detector_module

    class AnalysisDetector(FeatureRecordingMixin, detector_module.FallDetector):
        pass

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_path = args.output_dir / f"false_positive_analysis_{timestamp}.csv"
    alarm_samples = []
    false_alarm_videos = 0
    print(f"CSV: {log_path}", flush=True)
    with log_path.open("x", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for source in videos:
            try:
                events = analyze_video(source, args, AnalysisDetector, detector_module, cv2, writer)
            except (OSError, RuntimeError, ValueError) as error:
                raise SystemExit(f"Analysis incomplete: {error}\nPartial CSV: {log_path}") from error
            false_alarm_videos += bool(events)
            alarm_samples.extend(events)
            log_file.flush()
    print_summary(len(videos), false_alarm_videos, alarm_samples)
    print("CSV blanks mean unavailable observations/velocities, not zero motion.")
    print(f"CSV: {log_path}")


if __name__ == "__main__":
    main()
