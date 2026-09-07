import argparse
import re
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.fall_detector import FallDetector
from experiments.tracking_logger import TrackingLogger


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a tracking logging experiment")
    parser.add_argument("--model", default="yolov12n1.pt", help="Path to the YOLO model")
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index, video path, or image-sequence directory",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Pose/analysis confidence threshold")
    parser.add_argument("--detector-conf", type=float, default=0.3, help="YOLO tracker-input threshold")
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV preview window")
    return parser.parse_args()


def natural_sort_key(path):
    """Return a case-insensitive key that sorts numeric filename parts naturally."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def find_sequence_images(source_dir):
    """Find supported images directly inside a sequence directory."""
    return sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )


def display_frame(output_frame, no_display):
    """Display a processed frame and return whether the user requested a stop."""
    if no_display:
        return False

    cv2.imshow("Tracking Experiment", output_frame)
    return cv2.waitKey(1) & 0xFF == ord("q")


def process_image_sequence(source_dir, detector, tracking_logger, no_display):
    """Process a MOT-style directory of naturally sorted frame images."""
    image_paths = find_sequence_images(source_dir)
    if not image_paths:
        print(f"No JPG or PNG images found in: {source_dir}")
        return

    print(f"Image sequence: {source_dir} ({len(image_paths)} frames)")
    for sequence_index, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Could not read image, skipping: {image_path}")
            continue

        output_frame, _, fall_data = detector.process_frame(frame)
        frame_id = int(image_path.stem) if image_path.stem.isdigit() else sequence_index
        tracking_logger.log_frame(frame_id, fall_data)

        if display_frame(output_frame, no_display):
            break


def process_video(source_arg, detector, tracking_logger, no_display):
    """Process the existing camera or video-file input flow."""
    if source_arg.isdigit():
        cap = cv2.VideoCapture(int(source_arg), cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source_arg)

    if not cap.isOpened():
        print(f"Could not open video source: {source_arg}")
        return

    frame_id = 0
    try:
        while cap.isOpened():
            received, frame = cap.read()
            if not received:
                break

            output_frame, _, fall_data = detector.process_frame(frame)
            frame_id += 1
            tracking_logger.log_frame(frame_id, fall_data)

            if display_frame(output_frame, no_display):
                break
    finally:
        cap.release()


def main():
    args = parse_arguments()
    detector = FallDetector(
        model_path=args.model,
        confidence=args.conf,
        detector_confidence=args.detector_conf,
    )
    tracking_logger = TrackingLogger()

    print(f"Tracking log: {tracking_logger.log_path}")
    print("Press 'q' to stop the experiment.")
    try:
        source_path = Path(args.source)
        if source_path.is_dir():
            process_image_sequence(
                source_path, detector, tracking_logger, args.no_display
            )
        else:
            process_video(args.source, detector, tracking_logger, args.no_display)
    finally:
        detector.close()
        cv2.destroyAllWindows()

    print(f"Tracking log saved to: {tracking_logger.log_path}")


if __name__ == "__main__":
    main()
