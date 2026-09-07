"""Lifecycle simulations: no YOLO weights, real Pose inference, video or GUI.

Run: .venv/Scripts/python.exe -m unittest discover -s tests -v
"""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from models.transformer_fall_classifier import extract_transformer_features


ROOT = Path(__file__).resolve().parents[1]


def load_detector():
    """Import the production module with only external inference APIs mocked."""
    modules = {name: MagicMock() for name in (
        'cv2', 'mediapipe', 'torch', 'ultralytics',
        'ultralytics.engine.results', 'ultralytics.trackers.byte_tracker',
        'ultralytics.utils', 'ultralytics.utils.downloads',
        'models.transformer_fall_classifier',
    )}
    modules['torch'].__version__ = 'simulated'
    modules['torch'].cuda.is_available.return_value = False
    modules['ultralytics'].YOLO.return_value.parameters.side_effect = (
        lambda: iter([SimpleNamespace(device='cpu')])
    )
    modules['ultralytics.utils.downloads'].attempt_download.return_value = None
    modules['ultralytics.utils'].YAML.load.return_value = {}
    modules['ultralytics.utils'].IterableSimpleNamespace.side_effect = SimpleNamespace
    modules['cv2'].cvtColor.side_effect = lambda frame, mode: frame
    transformer_module = modules['models.transformer_fall_classifier']
    transformer_module.INPUT_TIMESTEPS = 30
    transformer_module.TRANSFORMER_INFERENCE_STRIDE = 3
    transformer_module.NUM_FEATURES = 51
    transformer_module.FALL_CONFIDENCE_THRESHOLD = 0.9
    transformer_module.extract_transformer_features.side_effect = extract_transformer_features
    spec = importlib.util.spec_from_file_location(
        'pose_lifecycle_detector', ROOT / 'models/fall_detector.py'
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', modules):
        spec.loader.exec_module(module)
    return module


def load_functions(path, names, namespace, class_name=None):
    """Execute unchanged application methods without loading Qt or showing windows."""
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef)
                     and n.name == class_name).body
    selected = [n for n in nodes if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class PoseLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.module = load_detector()
        with patch('builtins.print'):
            self.detector = self.module.FallDetector()
        self.created = []

        def factory(**kwargs):
            pose = MagicMock()
            pose.process.return_value = SimpleNamespace(pose_landmarks=None)
            self.created.append(pose)
            return pose

        self.factory = self.module.mp.solutions.pose.Pose
        self.factory.side_effect = factory
        self.detector.tracker.update.return_value = []
        self.frame = np.zeros((20, 20, 3), dtype=np.uint8)
        self.box = (0, 0, 20, 20)
        self.addCleanup(self.detector.close)

    def analyze(self, person_id):
        return self.detector.analyze_pose(self.frame, self.box, person_id)

    def test_lazy_creation_distinct_ids_and_cross_frame_reuse(self):
        self.factory.assert_not_called()
        self.analyze(1)
        self.analyze(2)
        self.analyze(np.int64(1))
        self.assertEqual(self.factory.call_count, 2)
        self.assertIsNot(self.detector.pose_estimators[1], self.detector.pose_estimators[2])
        self.assertEqual(self.created[0].process.call_count, 2)
        self.assertEqual(self.created[1].process.call_count, 1)
        self.factory.assert_called_with(
            static_image_mode=False, model_complexity=1, enable_segmentation=False,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )

    def test_invalid_ids_and_empty_crop_do_not_create_pose(self):
        for person_id in (None, 0, -1, True, np.bool_(True), 1.0, float('nan'), '1', []):
            with self.subTest(person_id=person_id):
                self.assertEqual(self.analyze(person_id), (None, None))
        self.detector.analyze_pose(self.frame, (0, 0, 0, 0), 1)
        self.factory.assert_not_called()

    def test_existing_smoothing_and_history_receive_person_id(self):
        self.analyze(7)
        landmark = SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9)
        self.created[0].process.return_value.pose_landmarks = SimpleNamespace(landmark=[landmark])
        with patch.object(self.detector, 'smooth_pose_landmarks', return_value=['smoothed']) as smooth:
            with patch.object(self.detector, 'calculate_pose_features', return_value={'angle': 0}) as features:
                self.assertEqual(self.analyze(7), (['smoothed'], {'angle': 0}))
        self.assertEqual(smooth.call_args.args[1], 7)
        features.assert_called_once_with(['smoothed'], 7)

    def test_one_euro_axes_people_and_raw_visibility_are_independent(self):
        first = [(0.1, 0.2, -0.3, 0.05)] * 33
        second = [(0.8, 0.9, -0.8, 0.01)] * 33
        self.detector.smooth_pose_landmarks(first, 1, 1.0)
        self.detector.smooth_pose_landmarks(first, 2, 1.0)
        before_other = self.detector.pose_filters[2][0]['x'].__dict__.copy()
        output = self.detector.smooth_pose_landmarks(second, 1, 1.1)
        self.assertEqual(self.detector.pose_filters[2][0]['x'].__dict__, before_other)
        for axis, index in (('x', 0), ('y', 1), ('z', 2)):
            reference = self.module.OneEuroFilter(**self.detector.pose_filter_params)
            reference(first[0][index], 1.0)
            self.assertEqual(output[0][index], reference(second[0][index], 1.1))
            self.assertNotEqual(output[0][index], first[0][index])
            self.assertIsNot(self.detector.pose_filters[1][0][axis],
                             self.detector.pose_filters[2][0][axis])
        self.assertEqual([point[3] for point in output], [0.01] * 33)
        self.assertEqual(len({id(f) for person in self.detector.pose_filters.values()
                              for axes in person.values() for f in axes.values()}), 198)
        self.detector.close()
        self.assertFalse(self.detector.pose_filters)

    def test_close_clears_auxiliary_state_without_tracker_or_pose_entry(self):
        self.detector.smooth_pose_landmarks([(0.1, 0.2, 0.3, 0.9)] * 33, 99, 1.0)
        self.detector.calculate_pose_features([(0.1, 0.2, 0.3, 0.9)] * 33, 99)
        self.assertNotIn(99, self.detector.person_states)
        self.detector.close()
        self.assertFalse(self.detector.pose_filters)
        self.assertFalse(self.detector.pose_histories)

    def test_expiry_closes_and_removes_pose_and_existing_state(self):
        for person_id, last_seen in ((1, 94.9), (2, 95.0)):
            self.analyze(person_id)
            self.detector.person_states[person_id] = {'last_seen': last_seen}
            self.detector.fallen_person_ids.add(person_id)
            self.detector.pose_histories[person_id].append({})
            self.detector.pose_filters[person_id] = {}
            self.detector.transformer_sequences[person_id].append(np.zeros(51))
            self.detector.transformer_probabilities[person_id] = 0.25
            self.detector.transformer_updates_since_inference[person_id] = 2
            self.detector.transformer_has_inferred.add(person_id)
            self.detector._raw_pose_landmarks[person_id] = ((0, 0, 0, 1),)
        with patch.object(self.module.time, 'time', return_value=100):
            self.detector.update_tracker(self.frame, [])
        self.created[0].close.assert_called_once()
        self.created[1].close.assert_not_called()
        for mapping in (self.detector.pose_estimators, self.detector.person_states,
                        self.detector.pose_histories, self.detector.pose_filters):
            self.assertNotIn(1, mapping)
            self.assertIn(2, mapping)
        self.assertNotIn(1, self.detector.transformer_sequences)
        self.assertNotIn(1, self.detector.transformer_probabilities)
        self.assertIn(2, self.detector.transformer_sequences)
        self.assertIn(2, self.detector.transformer_probabilities)
        for mapping in (self.detector.transformer_updates_since_inference,
                        self.detector.transformer_has_inferred,
                        self.detector._raw_pose_landmarks):
            self.assertNotIn(1, mapping)
            self.assertIn(2, mapping)
        self.assertNotIn(1, self.detector.fallen_person_ids)
        self.analyze(1)
        self.assertEqual(len(self.created), 3)

    def test_close_releases_all_is_repeatable_and_prevents_recreation(self):
        self.analyze(1)
        self.analyze(2)
        self.detector.transformer_sequences[1].append(np.ones(51, dtype=np.float32))
        self.detector.transformer_probabilities[1] = .95
        self.detector.transformer_updates_since_inference[1] = 2
        self.detector.transformer_has_inferred.add(1)
        self.detector._raw_pose_landmarks[1] = ((0, 0, 0, 1),)
        classifier = self.detector.transformer_classifier
        self.detector.close()
        self.detector.close()
        self.assertFalse(self.detector.pose_estimators)
        self.assertFalse(self.detector.transformer_sequences)
        self.assertFalse(self.detector.transformer_probabilities)
        self.assertIsNone(self.detector.transformer_classifier)
        self.assertFalse(self.detector.transformer_updates_since_inference)
        self.assertFalse(self.detector.transformer_has_inferred)
        self.assertFalse(self.detector._raw_pose_landmarks)
        classifier.close.assert_called_once()
        for pose in self.created:
            pose.close.assert_called_once()
        self.assertEqual(self.analyze(1), (None, None))
        self.assertEqual(len(self.created), 2)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.detector.process_frame(self.frame)

    def test_one_close_failure_does_not_skip_other_resources_and_can_retry(self):
        self.analyze(1)
        self.analyze(2)
        self.created[0].close.side_effect = [RuntimeError('simulated close failure'), None]
        with self.assertLogs(self.module.__name__, level='ERROR'):
            self.detector.close()
        self.created[1].close.assert_called_once()
        self.assertEqual(list(self.detector.pose_estimators), [1])
        self.detector.close()
        self.assertFalse(self.detector.pose_estimators)

    def test_close_and_expiry_wait_for_active_pose_processing(self):
        for operation in ('close', 'expire'):
            with self.subTest(operation=operation):
                if operation == 'expire':
                    # The first subcase closes its detector permanently.
                    self.setUp()
                self.analyze(1)
                pose = self.created[0]
                entered, release, attempting, finished = (threading.Event() for _ in range(4))
                errors = []

                def process(**kwargs):
                    entered.set()
                    if not release.wait(3):
                        raise TimeoutError('test did not release processing')
                    return SimpleNamespace(pose_landmarks=None)

                def analyze():
                    try:
                        self.analyze(1)
                    except BaseException as exc:
                        errors.append(exc)

                def cleanup():
                    attempting.set()
                    try:
                        if operation == 'close':
                            self.detector.close()
                        else:
                            self.detector.update_tracker(self.frame, [])
                    except BaseException as exc:
                        errors.append(exc)
                    finally:
                        finished.set()

                pose.process.side_effect = process
                self.detector.person_states[1] = {'last_seen': 0}
                worker = threading.Thread(target=analyze)
                closer = threading.Thread(target=cleanup)
                worker.start()
                try:
                    self.assertTrue(entered.wait(2))
                    closer.start()
                    self.assertTrue(attempting.wait(2))
                    self.assertFalse(finished.wait(0.1))
                    pose.close.assert_not_called()
                finally:
                    release.set()
                    worker.join(3)
                    if closer.ident is not None:
                        closer.join(3)
                self.assertFalse(worker.is_alive())
                self.assertFalse(closer.is_alive())
                self.assertEqual(errors, [])
                pose.close.assert_called_once()
                self.assertNotIn(1, self.detector.pose_estimators)


class ApplicationLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_functions(
            'dashboard/dashboard_app.py',
            {'stop_detection', 'start_detection', 'process_frame', 'toggle_detection'},
            dict(cv2=MagicMock(), QTimer=MagicMock(), QMessageBox=MagicMock(),
                 FallDetector=MagicMock(), os=SimpleNamespace(name='nt'), time=MagicMock()),
            class_name='FallDetectionDashboard',
        )
        dashboard_type = type('SimulatedDashboard', (), {
            name: self.namespace[name] for name in
            ('stop_detection', 'start_detection', 'process_frame', 'toggle_detection')
        })
        self.window = dashboard_type()
        self.window.__dict__.update(
            is_running=True, video_paused=False, video_timer=MagicMock(),
            cap=MagicMock(), fall_detector=MagicMock(), start_stop_btn=MagicMock(),
            pause_resume_btn=MagicMock(), status_bar=MagicMock(), video_source='test.mp4',
            camera_source=0, config=dict(model_path='unused.pt', confidence=0.5,
            ), tracked_fallen_ids={1},
            fallen_timestamps={1: 1}, detection_enabled=True, frame_count=0,
            last_fps_update=0, fps=0, fps_label=MagicMock(),
        )
        self.namespace['time'].time.return_value = 1

    def test_dashboard_stops_timer_before_close_and_is_repeatable(self):
        calls = MagicMock()
        detector, cap = self.window.fall_detector, self.window.cap
        calls.attach_mock(self.window.video_timer.stop, 'timer_stop')
        calls.attach_mock(detector.close, 'detector_close')
        self.window.stop_detection()
        self.assertEqual([call[0] for call in calls.mock_calls], ['timer_stop', 'detector_close'])
        self.window.stop_detection()
        detector.close.assert_called_once()
        cap.release.assert_called_once()
        self.assertIsNone(self.window.fall_detector)

    def test_dashboard_replacement_closes_previous_detector(self):
        old = self.window.fall_detector
        self.window.is_running = False
        self.window.start_detection()
        old.close.assert_called_once()
        self.assertIs(self.window.fall_detector, self.namespace['FallDetector'].return_value)
        self.assertTrue(self.window.is_running)
        self.assertFalse(self.window.fallen_timestamps)

    def test_dashboard_open_failure_releases_new_detector(self):
        self.window.is_running = False
        self.namespace['cv2'].VideoCapture.return_value.isOpened.return_value = False
        self.window.start_detection()
        self.namespace['FallDetector'].return_value.close.assert_called_once()
        self.assertIsNone(self.window.fall_detector)
        self.assertIsNone(self.window.cap)

    def test_dashboard_eof_stops_before_modal_message(self):
        self.window.cap.read.return_value = (False, None)
        detector = self.window.fall_detector
        self.namespace['QMessageBox'].information.side_effect = (
            lambda *args: self.assertFalse(self.window.is_running))
        self.window.process_frame()
        detector.close.assert_called_once()

    def test_dashboard_inference_error_releases_resources(self):
        detector = self.window.fall_detector
        self.window.cap.read.return_value = (True, np.zeros((2, 2, 3)))
        detector.process_frame.side_effect = RuntimeError('simulated inference error')
        self.window.process_frame()
        detector.close.assert_called_once()
        self.assertFalse(self.window.is_running)

    def test_cli_releases_on_open_failure_eof_and_processing_error(self):
        for outcome in ('open_failure', 'eof', 'processing_error'):
            with self.subTest(outcome=outcome):
                cv2, detector = MagicMock(), MagicMock()
                cap = cv2.VideoCapture.return_value
                cap.isOpened.return_value = outcome != 'open_failure'
                cap.read.return_value = (outcome == 'processing_error', object())
                detector.process_frame.side_effect = RuntimeError('simulated inference error')
                ns = load_functions('fall_detection_system.py', {'run_cli_mode'},
                                    {'FallDetector': MagicMock(return_value=detector)})
                args = SimpleNamespace(model='unused.pt', conf=0.5, source='test.mp4')
                with patch.dict('sys.modules', {'cv2': cv2, 'utils.utils': MagicMock()}):
                    with patch('builtins.print'), patch('traceback.print_exc'):
                        ns['run_cli_mode'](args)
                detector.close.assert_called_once()
                cap.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
