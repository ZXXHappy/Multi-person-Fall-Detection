"""Real CPU model/video validation; only GUI display is suppressed.

No synthetic probabilities, modified video frames, dependency installation,
batch resize, threshold changes, or replacement inference implementations.
"""
import json
import os
from pathlib import Path
import sys
from collections import defaultdict, deque
from types import SimpleNamespace
from unittest.mock import patch

os.environ['YOLO_AUTOINSTALL'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter
from models.fall_detector import FallDetector
from models.transformer_fall_classifier import extract_transformer_features
import fall_detection_system as cli
import utils.utils as utility

OUT = ROOT / 'validation_outputs' / 'transformer_cleanup_runtime'


def tensor_info(detail):
    return {key: detail[key].tolist() for key in ('shape', 'shape_signature')} | {
        'name': detail['name'], 'dtype': str(np.dtype(detail['dtype']))}


def independent_model():
    interpreter = Interpreter(model_content=(ROOT / 'models/fall_detection_transformer.tflite').read_bytes())
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    assert tuple(inp['shape']) == (1, 30, 51) and inp['dtype'] == np.float32
    assert tuple(out['shape']) == (1, 1) and out['dtype'] == np.float32
    values = np.linspace(-1, 1, 1530, dtype=np.float32).reshape(1, 30, 51)
    interpreter.set_tensor(inp['index'], values)
    interpreter.invoke()
    result = interpreter.get_tensor(out['index'])
    assert result.shape == (1, 1) and result.dtype == np.float32
    assert np.isfinite(result).all() and 0 <= result[0, 0] <= 1
    return {'input': tensor_info(inp), 'output': tensor_info(out),
            'synthetic_input': 'np.linspace(-1,1,1530,dtype=float32).reshape(1,30,51)',
            'real_model_probability': float(result[0, 0])}


class ObservedPose:
    def __init__(self, real, owner):
        self.real, self.owner = real, owner
        self.calls = self.closes = 0

    def process(self, **kwargs):
        assert not self.closes, 'Processing closed Pose'
        result = self.real.process(**kwargs)
        self.calls += 1
        self.owner.last_mp_raw = (tuple((p.x, p.y, p.z, p.visibility)
            for p in result.pose_landmarks.landmark) if result.pose_landmarks else None)
        return result

    def close(self):
        assert not self.closes, 'Pose closed twice'
        self.real.close()
        self.closes += 1


class ObservedDetector(FallDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.transformer_classifier.backend == 'ai_edge_litert.interpreter'
        assert self.device == 'cpu' and self.transformer_fall_threshold == 0.9
        self.rows, self.graphs = [], []
        self.counts, self.pose_counts = defaultdict(int), defaultdict(int)
        self.expected_sequences = defaultdict(lambda: deque(maxlen=30))
        self.calls = defaultdict(list)
        self.probabilities = defaultdict(list)
        self.graph_ids = {}
        self.raw_checks = self.manual_checks = self.filtered_differences = 0
        self.frame_index = self.invoke_count = 0
        self.errors = []
        self.classifier_reference = self.transformer_classifier
        real_factory = self.mp_pose.Pose

        def pose_factory(**params):
            graph = ObservedPose(real_factory(**params), self)
            self.graphs.append(graph)
            return graph

        self.mp_pose = SimpleNamespace(Pose=pose_factory, PoseLandmark=self.mp_pose.PoseLandmark)
        real_predict = self.transformer_classifier.predict

        def predict(sequence):
            values = np.asarray(sequence)
            assert values.shape == (30, 51) and values.dtype == np.float32
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(values, np.asarray(self.expected_sequences[self.current_id]))
            self.invoke_count += 1
            return real_predict(sequence)

        self.transformer_classifier.predict = predict

    def smooth_pose_landmarks(self, landmarks, person_id, timestamp):
        np.testing.assert_array_equal(landmarks, self.last_mp_raw)
        np.testing.assert_array_equal(self._raw_pose_landmarks[person_id], self.last_mp_raw)
        self.raw_checks += 1
        filtered = super().smooth_pose_landmarks(landmarks, person_id, timestamp)
        self.filtered_differences += not np.array_equal(filtered, landmarks)
        self.last_filtered = filtered
        return filtered

    def calculate_pose_features(self, landmarks, person_id):
        assert landmarks is self.last_filtered
        self.manual_checks += 1
        return super().calculate_pose_features(landmarks, person_id)

    def _analyze_pose(self, frame, person_box, person_id):
        self.current_box, self.current_shape = person_box, frame.shape
        self.last_mp_raw = None
        result = super()._analyze_pose(frame, person_box, person_id)
        graph = self.pose_estimators[person_id]
        previous = self.graph_ids.setdefault(person_id, id(graph))
        assert previous == id(graph), 'Pose not reused'
        assert len({id(g) for g in self.pose_estimators.values()}) == len(self.pose_estimators)
        if self.last_mp_raw is not None:
            np.testing.assert_array_equal(self.last_mp_raw, self._raw_pose_landmarks[person_id])
            self.pose_counts[person_id] += 1
        else:
            assert person_id not in self._raw_pose_landmarks
        return result

    def _update_transformer_sequence(self, person_id, features):
        self.current_id = person_id
        expected = extract_transformer_features(self.last_mp_raw, self.current_box, self.current_shape)
        np.testing.assert_array_equal(features, expected)
        self.expected_sequences[person_id].append(expected.copy())
        self.counts[person_id] += 1
        count = self.counts[person_id]
        before = self.invoke_count
        cached = self.transformer_probabilities.get(person_id)
        other_probabilities = {k: v for k, v in self.transformer_probabilities.items() if k != person_id}
        other_counters = {k: v for k, v in self.transformer_updates_since_inference.items() if k != person_id}
        probability, inferred = super()._update_transformer_sequence(person_id, features)
        expected_inferred = count >= 30 and (count - 30) % 3 == 0
        assert inferred == expected_inferred
        assert self.invoke_count - before == int(inferred)
        assert len(self.transformer_sequences[person_id]) == min(count, 30)
        np.testing.assert_array_equal(self.transformer_sequences[person_id], self.expected_sequences[person_id])
        assert len({id(s) for s in self.transformer_sequences.values()}) == len(self.transformer_sequences)
        assert other_probabilities == {k: v for k, v in self.transformer_probabilities.items() if k != person_id}
        assert other_counters == {k: v for k, v in self.transformer_updates_since_inference.items() if k != person_id}
        assert self.transformer_updates_since_inference[person_id] == (count if count < 30 else (count - 30) % 3)
        if inferred:
            self.calls[person_id].append({'video_frame': self.frame_index, 'person_update': count,
                                         'probability': probability})
            self.probabilities[person_id].append(probability)
        else:
            assert probability == cached
        return probability, inferred

    def process_frame(self, frame):
        self.frame_index += 1
        try:
            result = super().process_frame(frame)
            output, detected, data = result
            for person_id in data['person_ids']:
                prob = data['transformer_probabilities'][person_id]
                assert prob == self.transformer_probabilities.get(person_id)
                positive = prob is not None and prob >= 0.9
                assert (person_id in data['fallen_ids']) == positive
                assert (person_id in data['fall_event_ids']) == (positive and data['inferred_this_frame'][person_id])
            assert detected == bool(data['fallen_ids'])
            self.rows.append({'video_frame': self.frame_index, 'updates': dict(self.counts),
                              'counters': dict(self.transformer_updates_since_inference), 'fall_data': data})
            if self.frame_index % 20 == 0:
                print('\nREAL_VIDEO_PROGRESS', self.frame_index, dict(self.counts), flush=True)
            if self.frame_index in (30, 60):
                assert cv2.imwrite(str(OUT / f'annotated_{self.frame_index}.jpg'), output)
            return result
        except Exception as exc:
            self.errors.append(f'{type(exc).__name__}: {exc}')
            raise


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'independent_model': independent_model()}
    print('REAL_MODEL', json.dumps(report, ensure_ascii=True), flush=True)
    source = (ROOT / 'validation_outputs/source.txt').read_text(encoding='utf-8-sig').strip()
    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), source
    report['source'] = source
    report['video_metadata'] = {'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                                'frames': cap.get(cv2.CAP_PROP_FRAME_COUNT)}
    # Container frame-count metadata can differ from decodable frames.
    decoded_frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        decoded_frames += 1
    report['independently_decoded_frames'] = decoded_frames
    cap.release()
    detectors, captures, snapshots = [], [], []
    real_capture, real_save = cv2.VideoCapture, utility.save_frame

    def factory(**kwargs):
        detector = ObservedDetector(**kwargs)
        detectors.append(detector)
        return detector

    def capture(*args, **kwargs):
        cap = real_capture(*args, **kwargs)
        captures.append(cap)
        return cap

    def save(*args, **kwargs):
        path = real_save(*args, **kwargs)
        assert path and Path(path).is_file()
        snapshots.append(str(path))
        return path

    args = SimpleNamespace(model=str(ROOT / 'yolov12n1.pt'), conf=0.5, source=source,
                           save_falls=True, output_dir=str(OUT / 'alarm_snapshots'))
    # Run the actual CLI loop and alarm/snapshot consumer to natural EOF.
    # Only window display and keyboard polling are suppressed for headless execution.
    with patch.object(cli, 'FallDetector', factory), patch.object(cv2, 'VideoCapture', capture), \
         patch.object(cv2, 'imshow', lambda *a: None), patch.object(cv2, 'waitKey', lambda *a: -1), \
         patch.object(utility, 'save_frame', save):
        cli.run_cli_mode(args)
    detector = detectors[0]
    detector.close()  # Confirm repeated close remains safe.
    report['frames_processed'] = len(detector.rows)
    report['people'] = {pid: {'updates': count, 'pose_outputs': detector.pose_counts[pid],
        'missing_pose_zero_vectors': count - detector.pose_counts[pid],
        'collected_30': count >= 30, 'inferences': detector.calls[pid],
        'probability_min': min(detector.probabilities[pid]) if detector.probabilities[pid] else None,
        'probability_max': max(detector.probabilities[pid]) if detector.probabilities[pid] else None}
        for pid, count in detector.counts.items()}
    report['raw_branch_checks'] = detector.raw_checks
    report['manual_filtered_branch_checks'] = detector.manual_checks
    report['frames_with_filter_difference'] = detector.filtered_differences
    report['real_video_invokes'] = detector.invoke_count
    report['event_candidates'] = [{'video_frame': r['video_frame'], 'ids': r['fall_data']['fall_event_ids']}
                                  for r in detector.rows if r['fall_data']['fall_event_ids']]
    report['cli_alarm_snapshots'] = snapshots
    report['errors'] = detector.errors
    empty_names = ('pose_estimators', 'pose_filters', 'pose_histories',
                   'transformer_sequences', 'transformer_probabilities',
                   'transformer_updates_since_inference', 'transformer_has_inferred',
                   '_raw_pose_landmarks', 'person_states', 'fallen_person_ids')
    report['cleanup'] = {name: not bool(getattr(detector, name)) for name in empty_names}
    report['cleanup']['interpreter_released'] = detector.classifier_reference.interpreter is None
    report['cleanup']['classifier_released'] = detector.transformer_classifier is None
    report['cleanup']['captures_released'] = all(not cap.isOpened() for cap in captures)
    report['cleanup']['pose_close_exactly_once'] = all(g.closes == 1 for g in detector.graphs)
    report['frames'] = detector.rows
    baseline_path = ROOT / 'validation_outputs/transformer_real_runtime/results.json'
    if baseline_path.is_file():
        baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
        differences = {}
        assert {str(k) for k in report['people']} == set(baseline['people'])
        for person_id, person in report['people'].items():
            old = baseline['people'][str(person_id)]
            assert person['updates'] == old['updates']
            assert [r['person_update'] for r in person['inferences']] == [
                r['person_update'] for r in old['inferences']]
            current = np.array([r['probability'] for r in person['inferences']])
            previous = np.array([r['probability'] for r in old['inferences']])
            np.testing.assert_allclose(current, previous, rtol=0, atol=1e-6)
            differences[person_id] = float(np.max(np.abs(current - previous)))
        report['baseline_comparison'] = {'source': str(baseline_path),
                                         'max_absolute_probability_difference': differences}
    (OUT / 'results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    assert not report['errors'] and all(report['cleanup'].values())
    assert report['frames_processed'] == report['independently_decoded_frames']
    assert len(report['people']) >= 2 and all(p['collected_30'] for p in report['people'].values())
    print('\nREAL_VALIDATION_OK', json.dumps({k: v for k, v in report.items() if k != 'frames'}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
