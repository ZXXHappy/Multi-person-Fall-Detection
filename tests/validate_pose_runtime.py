"""Real local inference checks; no downloads or dependency installation.

The lifecycle scenario masks a person in a real source frame, then restores it.
It uses real wall time, YOLO, ByteTrack and MediaPipe (no mocked IDs or clocks).
The benchmark uses unmodified video frames and an experiment-only shared baseline.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import mediapipe as mp
import numpy as np
import torch
from models.fall_detector import FallDetector

OUT = ROOT / 'validation_outputs'
REAL_POSE = mp.solutions.pose.Pose


class MeasuredPose:
    def __init__(self, owner, **kwargs):
        self.serial = len(owner.graphs) + 1
        self.closed = False
        self.calls = self.successes = self.closes = 0
        self.seconds = 0.0
        start = time.perf_counter()
        self.inner = REAL_POSE(**kwargs)
        self.creation_seconds = time.perf_counter() - start
        owner.graphs.append(self)

    def process(self, **kwargs):
        if self.closed:
            raise AssertionError('Attempt to process a closed Pose graph')
        start = time.perf_counter()
        result = self.inner.process(**kwargs)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        self.successes += bool(result.pose_landmarks)
        return result

    def close(self):
        if self.closed:
            raise AssertionError('Pose graph closed twice')
        self.inner.close()
        self.closed = True
        self.closes += 1


class RecordedDetector(FallDetector):
    def __init__(self, shared=False):
        super().__init__()
        self.graphs = []
        self.shared = shared
        self.shared_pose = None
        self.observations = []
        self.mp_pose = SimpleNamespace(
            Pose=lambda **kw: MeasuredPose(self, **kw),
            PoseLandmark=mp.solutions.pose.PoseLandmark,
        )

    def _analyze_pose(self, frame, box, person_id):
        if self.shared:
            # Reproduce only the old shared temporal graph, keeping all other
            # detector, tracker, filter and alarm settings identical.
            if self.shared_pose is not None:
                self.pose_estimators[person_id] = self.shared_pose
        result = super()._analyze_pose(frame, box, person_id)
        graph = self.pose_estimators.get(person_id)
        if self.shared and graph is not None:
            self.shared_pose = graph
        self.observations.append(dict(
            person_id=person_id, graph=graph.serial if graph else None,
            pose=bool(result[0]), features=bool(result[1]),
        ))
        return result

    def _close_pose_estimator(self, person_id):
        if self.shared:
            self.pose_estimators.pop(person_id, None)
        else:
            super()._close_pose_estimator(person_id)

    def close(self):
        super().close()
        if self.shared_pose is not None and not self.shared_pose.closed:
            self.shared_pose.close()


def process(detector, frame, phase, rows):
    detector.observations = []
    start = time.perf_counter()
    _, _, data = detector.process_frame(frame)
    elapsed = time.perf_counter() - start
    rows.append(dict(
        phase=phase, frame=len(rows), seconds=elapsed,
        ids=data['person_ids'], observations=list(detector.observations),
        graphs={k: v.serial for k, v in detector.pose_estimators.items()},
    ))
    return data


def assert_closed(detector):
    assert not detector.pose_estimators
    assert all(g.closed and g.closes == 1 for g in detector.graphs)


def lifecycle():
    frame = cv2.imread(str(OUT / 'candidate.jpg'))
    assert frame is not None
    rows = []
    detector = RecordedDetector()
    try:
        for _ in range(20):
            data = process(detector, frame, 'both', rows)
        assert len(data['person_ids']) == 2, data
        original = dict(detector.pose_estimators)
        assert len(set(id(g) for g in original.values())) == 2
        boxes = list(zip(data['person_ids'], data['person_boxes']))
        leaving, box = min(boxes, key=lambda item: (item[1][0] + item[1][2]) / 2)
        staying = next(i for i in original if i != leaving)
        # Remove the left person's head/body while retaining the right face.
        hidden = frame.copy()
        cutoff = int(frame.shape[1] * 0.30)
        hidden[:, :cutoff] = 0
        cv2.imwrite(str(OUT / 'controlled_absence.jpg'), hidden)
        start = time.monotonic()
        count = 0
        while time.monotonic() - start <= 6.5 or count < 45:
            process(detector, hidden, 'left_person_hidden', rows)
            count += 1
        elapsed = time.monotonic() - start
        assert leaving not in detector.pose_estimators, ('not expired', leaving)
        assert original[leaving].closed
        assert detector.pose_estimators[staying] is original[staying]
        assert not original[staying].closed
        for _ in range(25):
            data = process(detector, frame, 'both_restored', rows)
        assert len(data['person_ids']) == 2, data
        assert detector.pose_estimators[staying] is original[staying]
        returned = next(i for i in data['person_ids'] if i != staying)
        assert detector.pose_estimators[returned] is not original[leaving]
        result = dict(
            scenario='controlled masking of a real video frame; real inference and wall clock',
            leaving_id=leaving, staying_id=staying, returning_id=returned,
            absence_seconds=elapsed, absence_frames=count, rows=rows,
        )
    finally:
        detector.close()
        detector.close()
    assert_closed(detector)
    result['all_graphs_closed_once'] = True
    (OUT / 'lifecycle.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('LIFECYCLE', {k: v for k, v in result.items() if k != 'rows'}, flush=True)
    return result


def benchmark():
    source = Path((OUT / 'source.txt').read_text(encoding='utf-8'))
    cap = cv2.VideoCapture(str(source))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    assert len(frames) > 20
    runs = []
    # ABBA ordering reduces a simple warm-cache/order bias.
    for run, shared in enumerate((True, False, False, True), 1):
        detector = RecordedDetector(shared=shared)
        rows = []
        try:
            for frame in frames:
                process(detector, frame, 'shared' if shared else 'isolated', rows)
        finally:
            detector.close()
        assert_closed(detector)
        measured = rows[10:]
        seconds = sum(r['seconds'] for r in measured)
        multi = [r for r in measured if len(r['observations']) >= 2]
        successful = sum(all(o['pose'] for o in r['observations']) for r in multi)
        result = dict(
            run=run, mode='shared' if shared else 'isolated',
            frames=len(frames), warmup_frames=10, measured_frames=len(measured),
            fps=len(measured)/seconds, mean_ms=1000*seconds/len(measured),
            p95_ms=float(np.percentile([r['seconds']*1000 for r in measured], 95)),
            two_person_frames=len(multi), both_pose_output_frames=successful,
            first_frame_ms=rows[0]['seconds']*1000,
            graph_count=len(detector.graphs), rows=rows,
        )
        runs.append(result)
        print('BENCHMARK', {k: v for k, v in result.items() if k != 'rows'}, flush=True)
    result = dict(
        source=str(source), source_fps=source_fps, resolution=list(frames[0].shape[:2]),
        python=sys.executable, torch=torch.__version__, mediapipe=mp.__version__,
        cuda=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
        timing='process_frame only; excludes decode/display, model load and first 10 frames',
        runs=runs,
    )
    (OUT / 'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with (OUT / 'benchmark.csv').open('w', newline='', encoding='utf-8') as f:
        keys = [k for k in runs[0] if k != 'rows']
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in keys} for r in runs)
    return result


def application_restart():
    # Run Qt offscreen: real widgets and signals; no visible desktop window.
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from PyQt5.QtWidgets import QApplication
    import dashboard.dashboard_app as dashboard
    import fall_detection_system as cli

    source = (OUT / 'source.txt').read_text(encoding='utf-8')
    created = []

    def create(**kwargs):
        detector = RecordedDetector()
        created.append(detector)
        return detector

    args = SimpleNamespace(model='yolov12n1.pt', conf=.5, source=source,
                           save_falls=False, output_dir=str(OUT))
    # Real CLI loop and inference; suppress only the OpenCV display/key UI.
    with patch.object(cli, 'FallDetector', side_effect=create), \
            patch.object(cv2, 'imshow'), patch.object(cv2, 'waitKey', return_value=ord('q')):
        for _ in range(2):
            cli.run_cli_mode(args)
            assert created[-1].graphs
            assert_closed(created[-1])

    app = QApplication.instance() or QApplication([])
    window = dashboard.FallDetectionDashboard()
    window.video_source = source
    window.config['sound_alerts'] = False
    window.config['auto_save_falls'] = False
    errors = []
    with patch.object(dashboard, 'FallDetector', side_effect=create), \
            patch.object(dashboard.QMessageBox, 'critical', side_effect=lambda *a: errors.append(str(a))):
        for _ in range(2):
            window.start_detection()
            assert window.is_running
            # Drive the GUI event loop so the actual QTimer invokes inference.
            deadline = time.monotonic() + 15
            while sum(g.calls for g in created[-1].graphs) < 8 and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(.005)
            assert sum(g.calls for g in created[-1].graphs) >= 8
            window.stop_detection()
            assert not window.video_timer.isActive()
            assert_closed(created[-1])
        window.start_detection()
        window.process_frame()
        window.close()  # Actual Qt closeEvent cleanup.
        assert_closed(created[-1])
        app.aboutToQuit.emit()  # Repeat application-exit hook safely.
    assert not errors, errors
    result = dict(cli_restarts=2, dashboard_restarts=2, close_event=True,
                  repeated_exit_hook=True, detectors=len(created), errors=errors,
                  caveat='Qt offscreen; CLI display/key handling suppressed; real video/inference')
    (OUT / 'application_restart.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('APPLICATION_RESTART', result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('check', choices=('lifecycle', 'benchmark', 'applications'))
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    {'lifecycle': lifecycle, 'benchmark': benchmark,
     'applications': application_restart}[args.check]()
