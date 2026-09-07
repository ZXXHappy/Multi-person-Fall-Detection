import csv
import datetime
from pathlib import Path


class TrackingLogger:
    """Write tracking results from ``FallDetector.process_frame`` to CSV."""

    HEADER = [
        "frame_id",
        "person_id",
        "bbox_center_x",
        "bbox_center_y",
        "bbox_width",
        "bbox_height",
    ]

    def __init__(self, output_dir=None):
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent / "logs"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.log_path = output_dir / f"tracking_{timestamp}.csv"

        with self.log_path.open("w", newline="", encoding="utf-8") as log_file:
            csv.writer(log_file).writerow(self.HEADER)

    def log_frame(self, frame_id, fall_data):
        """Record the persons exposed through the detector's public result data."""
        rows = []

        person_ids = fall_data["person_ids"]
        person_boxes = fall_data["person_boxes"]
        for person_id, box in zip(person_ids, person_boxes):
            x1, y1, x2, y2 = box
            bbox_width = float(x2 - x1)
            bbox_height = float(y2 - y1)
            bbox_center_x = float(x1 + bbox_width / 2)
            bbox_center_y = float(y1 + bbox_height / 2)

            rows.append([
                frame_id,
                person_id,
                bbox_center_x,
                bbox_center_y,
                bbox_width,
                bbox_height,
            ])

        with self.log_path.open("a", newline="", encoding="utf-8") as log_file:
            writer = csv.writer(log_file)
            if rows:
                writer.writerows(rows)
            else:
                writer.writerow([frame_id, "", "", "", "", ""])
