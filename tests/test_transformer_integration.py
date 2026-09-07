import ast
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from models.transformer_fall_classifier import (
    INPUT_TIMESTEPS,
    NUM_FEATURES,
    TRANSFORMER_KEYPOINT_INDICES,
    TRANSFORMER_KEYPOINT_NAMES,
    TransformerFallClassifier,
    TFLiteUnavailableError,
    extract_transformer_features,
    normalize_skeleton_frame,
)
from test_pose_lifecycle import load_detector, load_functions


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DEPLOYMENT = (
    ROOT.parent / "Fall-Detection" / "deployment" / "raspberry_pi" / "fall-detector.py"
)


def upstream_normalizer():
    tree = ast.parse(UPSTREAM_DEPLOYMENT.read_text(encoding="utf-8"))
    names = {"get_kpt_indices_training_order", "normalize_skeleton_frame"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "np": np,
        "MIN_KEYPOINT_CONFIDENCE_FOR_NORMALIZATION": 0.3,
        "SORTED_YOUR_KEYPOINT_NAMES": list(TRANSFORMER_KEYPOINT_NAMES),
        "KEYPOINT_DICT_TRAINING": {
            name: index for index, name in enumerate(TRANSFORMER_KEYPOINT_NAMES)
        },
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(UPSTREAM_DEPLOYMENT), "exec"), namespace)
    return namespace["normalize_skeleton_frame"]


class FakeInterpreter:
    output_probability = np.float32(0.93)

    def __init__(self, model_content):
        self.model_content = model_content
        self.allocated = False
        self.input = None
        self.invoked = False

    def allocate_tensors(self):
        self.allocated = True

    def get_input_details(self):
        return [{"shape": np.array([1, 30, 51]), "dtype": np.float32, "index": 4}]

    def get_output_details(self):
        return [{"shape": np.array([1, 1]), "dtype": np.float32, "index": 9}]

    def set_tensor(self, index, value):
        assert index == 4
        self.input = value

    def invoke(self):
        self.invoked = True

    def get_tensor(self, index):
        assert index == 9
        return np.array([[self.output_probability]], dtype=np.float32)


class TransformerFeatureTests(unittest.TestCase):
    def test_published_keypoint_order_and_xyz_visibility_layout(self):
        self.assertEqual(
            TRANSFORMER_KEYPOINT_INDICES,
            (27, 7, 13, 2, 23, 25, 11, 15, 0,
             28, 8, 14, 5, 24, 26, 12, 16),
        )
        landmarks = [
            (index / 100.0, index / 200.0, -index, 0.31 + index / 1000.0)
            for index in range(33)
        ]
        box = (20, 40, 120, 240)
        frame_shape = (400, 200, 3)
        raw = np.zeros(51, dtype=np.float32)
        for position, landmark_index in enumerate(TRANSFORMER_KEYPOINT_INDICES):
            x, y, _, visibility = landmarks[landmark_index]
            raw[position * 3:position * 3 + 3] = (
                (20 + x * 100) / 200,
                (40 + y * 200) / 400,
                visibility,
            )
        expected = upstream_normalizer()(raw.copy())
        actual = extract_transformer_features(landmarks, box, frame_shape)
        self.assertEqual(actual.shape, (51,))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        np.testing.assert_array_equal(actual[2::3], raw[2::3])

    def test_normalization_matches_upstream_edge_cases(self):
        upstream = upstream_normalizer()
        cases = []

        regular = np.arange(51, dtype=np.float32) / 100
        regular[20] = regular[47] = regular[14] = regular[41] = 0.9
        cases.append(regular)

        one_sided = regular.copy()
        one_sided[47] = one_sided[41] = 0.3  # strict > 0.3 means invalid
        cases.append(one_sided)

        no_hip = regular.copy()
        no_hip[14] = no_hip[41] = 0.0
        cases.append(no_hip)

        tiny_scale = regular.copy()
        tiny_scale[19] = tiny_scale[13] + np.float32(1e-6)
        tiny_scale[46] = tiny_scale[40] + np.float32(1e-6)
        cases.append(tiny_scale)

        no_shoulder = regular.copy()
        no_shoulder[[20, 47]] = 0
        cases.append(no_shoulder)

        right_only = regular.copy()
        right_only[[20, 14]] = 0
        cases.append(right_only)

        for features in cases:
            with self.subTest(features=features.tolist()):
                expected = upstream(features.copy())
                actual = normalize_skeleton_frame(features.copy())
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
                np.testing.assert_array_equal(actual[2::3], features[2::3])

    def test_empty_pose_produces_zero_vector(self):
        actual = extract_transformer_features(None, (0, 0, 10, 10), (10, 10, 3))
        np.testing.assert_array_equal(actual, np.zeros(51, dtype=np.float32))

    def test_nonfinite_features_are_sanitized_after_normalization(self):
        # No valid hip references: the original coordinates would be returned.
        data = np.zeros(51, dtype=np.float32)
        data[0], data[1], data[2] = np.nan, np.inf, -np.inf
        actual = normalize_skeleton_frame(data)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual, np.zeros(51, dtype=np.float32))

    def test_crop_coordinates_are_restored_even_without_valid_hip_references(self):
        landmarks = [(0.25, 0.75, -100.0, 0.2)] * 33
        actual = extract_transformer_features(landmarks, (20, 40, 120, 240), (400, 200, 3))
        expected = np.tile(np.array([.225, .475, .2], dtype=np.float32), 17)
        np.testing.assert_array_equal(actual, expected)


class TFLiteWrapperTests(unittest.TestCase):
    def test_backend_falls_through_after_model_allocation_failure(self):
        class BrokenInterpreter(FakeInterpreter):
            def allocate_tensors(self):
                raise RuntimeError("unsupported model operation")
        with patch("models.transformer_fall_classifier.import_module") as importer:
            importer.side_effect = [SimpleNamespace(Interpreter=BrokenInterpreter),
                                    SimpleNamespace(Interpreter=FakeInterpreter)]
            classifier = TransformerFallClassifier()
        self.assertEqual(classifier.backend, "tflite_runtime.interpreter")
        self.assertIn("unsupported model operation", classifier.load_errors[0])
        self.assertEqual(importer.call_count, 2)
        classifier.close()

    def test_unavailable_backends_report_each_import_error(self):
        with patch("models.transformer_fall_classifier.import_module",
                   side_effect=ModuleNotFoundError("simulated missing package")):
            with self.assertRaises(TFLiteUnavailableError) as caught:
                TransformerFallClassifier()
        for name in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow"):
            self.assertIn(name, str(caught.exception))

    def test_tensorflow_public_lite_api_is_supported(self):
        with patch("models.transformer_fall_classifier.import_module") as importer:
            importer.side_effect = [ModuleNotFoundError("no litert"),
                                    ModuleNotFoundError("no tflite_runtime"),
                                    SimpleNamespace(lite=SimpleNamespace(Interpreter=FakeInterpreter))]
            classifier = TransformerFallClassifier()
        self.assertEqual(classifier.backend, "tensorflow")
        classifier.close()

    def test_model_contract_rejects_wrong_shape_or_dtype(self):
        for target, field, value in (
            ("input", "shape", np.array([10, 30, 51])),
            ("input", "dtype", np.int8),
            ("output", "shape", np.array([1, 2])),
            ("output", "dtype", np.int8),
        ):
            with self.subTest(target=target, field=field):
                method = f"get_{target}_details"
                detail = getattr(FakeInterpreter("unused"), method)()[0]
                detail[field] = value
                with patch.object(FakeInterpreter, method, return_value=[detail]):
                    with self.assertRaises(ValueError):
                        TransformerFallClassifier(interpreter_class=FakeInterpreter)

    def test_default_model_path_is_independent_of_working_directory(self):
        # Changing cwd's reported value does not affect __file__-based resolution.
        with patch("os.getcwd", return_value="C:\\Windows"):
            classifier = TransformerFallClassifier(interpreter_class=FakeInterpreter)
        self.assertEqual(classifier.model_path, ROOT / "models/fall_detection_transformer.tflite")
        self.assertEqual(classifier.interpreter.model_content, classifier.model_path.read_bytes())
        classifier.close()

    def test_model_input_shape_dtype_and_invocation(self):
        classifier = TransformerFallClassifier(
            model_path=ROOT / "models" / "fall_detection_transformer.tflite",
            interpreter_class=FakeInterpreter,
        )
        probability = classifier.predict(
            deque((np.full(51, index, dtype=np.float32) for index in range(30)), maxlen=30)
        )
        self.assertAlmostEqual(probability, 0.93, places=6)
        self.assertEqual(classifier.interpreter.input.shape, (1, 30, 51))
        self.assertEqual(classifier.interpreter.input.dtype, np.float32)
        self.assertTrue(classifier.interpreter.invoked)
        classifier.close()
        classifier.close()
        self.assertIsNone(classifier.interpreter)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            classifier.predict(np.zeros((30, 51), dtype=np.float32))


class TransformerPipelineTests(unittest.TestCase):
    def setUp(self):
        self.module = load_detector()
        with patch("builtins.print"):
            self.detector = self.module.FallDetector()
        self.classifier = self.detector.transformer_classifier
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.boxes = [(0, 0, 80, 100), (100, 0, 200, 100)]
        self.detector.detect_person = MagicMock(return_value=[])
        self.detector.update_tracker = MagicMock(
            return_value=(self.boxes, {0: 1, 1: 2})
        )
        landmarks = [(0.5, 0.5, 0.0, 0.9)] * 33
        def fake_analyze(frame, box, person_id):
            if person_id == 1:
                self.detector._raw_pose_landmarks[person_id] = tuple(landmarks)
                return landmarks, {"angle": 90}
            return None, None
        self.detector.analyze_pose = MagicMock(side_effect=fake_analyze)
        self.classifier.predict.side_effect = (
            lambda sequence: 0.95 if np.any(np.asarray(sequence)) else 0.1
        )
        self.addCleanup(self.detector.close)

    def test_sequences_are_per_id_missing_pose_is_zero_and_transformer_alarms(self):
        for _ in range(30):
            _, fall_detected, fall_data = self.detector.process_frame(self.frame)

        self.assertIsNot(
            self.detector.transformer_sequences[1],
            self.detector.transformer_sequences[2],
        )
        self.assertEqual(len(self.detector.transformer_sequences[1]), 30)
        self.assertEqual(len(self.detector.transformer_sequences[2]), 30)
        self.assertTrue(np.any(self.detector.transformer_sequences[1][-1]))
        np.testing.assert_array_equal(
            self.detector.transformer_sequences[2][-1],
            np.zeros(NUM_FEATURES, dtype=np.float32),
        )
        self.assertEqual(self.classifier.predict.call_count, 2)
        for call in self.classifier.predict.call_args_list:
            model_input = np.asarray(call.args[0], dtype=np.float32)
            self.assertEqual(model_input.shape, (30, 51))
            self.assertEqual(model_input.dtype, np.float32)
        self.assertTrue(fall_detected)
        self.assertEqual(fall_data["fallen_ids"], [1])
        self.assertEqual(fall_data["fall_type"], "transformer_fall")
        self.assertEqual(fall_data["transformer_probabilities"], {1: 0.95, 2: 0.1})
        self.assertEqual(fall_data["transformer_sequence_lengths"], {1: 30, 2: 30})

    def test_clipped_box_is_used_for_crop_and_full_frame_restoration(self):
        self.detector.update_tracker.return_value = (
            [(-10, -20, 220, 130)], {0: 1}
        )
        self.detector.process_frame(self.frame)
        passed_box = self.detector.analyze_pose.call_args.args[1]
        self.assertEqual(passed_box, (0, 0, 200, 100))
        feature_box = self.module.extract_transformer_features.call_args.args[1]
        self.assertEqual(feature_box, (0, 0, 200, 100))

    def test_collecting_window_never_calls_old_rule_alarm(self):
        _, detected, data = self.detector.process_frame(self.frame)
        self.assertFalse(detected)
        self.assertEqual(data["fallen_ids"], [])
        self.assertEqual(data["transformer_probabilities"], {1: None, 2: None})

    def test_independent_stride_counts_include_missing_pose_updates(self):
        inferred_at = {1: [], 2: []}
        for frame_number in range(1, 44):
            if frame_number <= 10:
                self.detector.update_tracker.return_value = ([self.boxes[0]], {0: 1})
            else:
                self.detector.update_tracker.return_value = (self.boxes, {0: 1, 1: 2})
            before = self.classifier.predict.call_count
            _, _, data = self.detector.process_frame(self.frame)
            for person_id, inferred in data["inferred_this_frame"].items():
                if inferred:
                    inferred_at[person_id].append(frame_number)
                    self.assertEqual(self.detector.transformer_updates_since_inference[person_id], 0)
            self.assertEqual(self.classifier.predict.call_count - before,
                             sum(data["inferred_this_frame"].values()))
            if frame_number < 30:
                self.assertIsNone(data["transformer_probabilities"][1])
                self.assertFalse(data["transformer_has_inferred"][1])
            else:
                self.assertEqual(data["transformer_probabilities"][1], .95)
                self.assertTrue(data["transformer_has_inferred"][1])
                self.assertEqual(data["fall_event_ids"],
                                 [1] if data["inferred_this_frame"][1] else [])
        self.assertEqual(inferred_at, {1: [30, 33, 36, 39, 42], 2: [40, 43]})
        self.assertEqual(len(self.detector.transformer_sequences[2]), 30)
        np.testing.assert_array_equal(np.asarray(self.detector.transformer_sequences[2]),
                                      np.zeros((30, 51), dtype=np.float32))

    def test_every_element_is_buffered_and_latest_window_used(self):
        snapshots = []
        self.classifier.predict.side_effect = lambda sequence: (
            snapshots.append(np.asarray(sequence).copy()) or .2)
        for update in range(1, 40):
            feature = np.full(51, update, dtype=np.float32)
            self.detector._update_transformer_sequence(9, feature)
        self.assertEqual(len(snapshots), 4)
        for snapshot, first, last in zip(snapshots, [1, 4, 7, 10], [30, 33, 36, 39]):
            np.testing.assert_array_equal(snapshot[:, 0], np.arange(first, last + 1))

    def test_new_low_probability_clears_alarm_after_cached_positive_frames(self):
        self.detector.update_tracker.return_value = ([self.boxes[0]], {0: 1})
        self.classifier.predict.side_effect = [.95, .1]
        for _ in range(30):
            _, _, data = self.detector.process_frame(self.frame)
        self.assertEqual(data["fall_event_ids"], [1])
        for _ in range(2):
            _, detected, data = self.detector.process_frame(self.frame)
            self.assertTrue(detected)
            self.assertEqual(data["fallen_ids"], [1])
            self.assertEqual(data["fall_event_ids"], [])
            self.assertEqual(data["transformer_probabilities"][1], .95)
        _, detected, data = self.detector.process_frame(self.frame)
        self.assertFalse(detected)
        self.assertEqual(data["fallen_ids"], [])
        self.assertNotIn(1, self.detector.fallen_person_ids)
        self.assertEqual(self.classifier.predict.call_count, 2)

    def test_invalid_boxes_and_ids_skip_pose_and_sequence_updates(self):
        self.detector.update_tracker.return_value = (
            [(200, 0, 230, 30), self.boxes[0], self.boxes[1]], {0: 1, 1: None, 2: -2})
        _, _, data = self.detector.process_frame(self.frame)
        self.detector.analyze_pose.assert_not_called()
        self.assertEqual(data["person_count"], 0)
        self.assertEqual(data["person_ids"], [])
        self.assertFalse(self.detector.transformer_sequences)
        self.assertFalse(self.detector.transformer_updates_since_inference)


class RawPoseBranchTests(unittest.TestCase):
    def test_transformer_uses_raw_pose_while_manual_features_use_filtered_pose(self):
        for visibility in (.9, .2):
            with self.subTest(visibility=visibility):
                module = load_detector()
                with patch("builtins.print"):
                    detector = module.FallDetector()
                self.addCleanup(detector.close)
                frame = np.zeros((100, 200, 3), dtype=np.uint8)
                box = (0, 10, 150, 100)
                detector.detect_person = MagicMock(return_value=[])
                detector.update_tracker = MagicMock(return_value=([(-20, 10, 150, 120)], {0: 7}))
                seed = [(0.2 + i / 100, 0.2 + i / 100, 0.0, .9) for i in range(33)]
                changed = list(seed)
                changed[15] = (.95, .85, -.3, visibility)
                def result(points):
                    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=[
                        SimpleNamespace(x=x, y=y, z=z, visibility=v) for x, y, z, v in points]))
                pose = module.mp.solutions.pose.Pose.return_value
                pose.process.side_effect = [
                    result(seed), result(changed), SimpleNamespace(pose_landmarks=None)]
                detector.calculate_pose_features = MagicMock(return_value={"angle": 0})
                with patch.object(module.time, "monotonic", side_effect=[1., 1.1]):
                    detector.process_frame(frame)
                    detector.process_frame(frame)
                raw = detector._raw_pose_landmarks[7]
                smoothed = detector.calculate_pose_features.call_args.args[0]
                self.assertEqual(raw, tuple(changed))
                self.assertNotEqual(smoothed[15][0], raw[15][0])
                self.assertEqual(smoothed[15][3], raw[15][3])
                self.assertNotEqual(smoothed[15][:3], seed[15][:3])
                np.testing.assert_array_equal(detector.transformer_sequences[7][-1],
                                              extract_transformer_features(raw, box, frame.shape))
                self.assertFalse(np.allclose(detector.transformer_sequences[7][-1],
                                             extract_transformer_features(smoothed, box, frame.shape)))
                # The actual pose input uses exactly the clipped dimensions.
                self.assertEqual(pose.process.call_args.kwargs["image"].shape, (90, 150, 3))
                detector.process_frame(frame)
                self.assertNotIn(7, detector._raw_pose_landmarks)
                np.testing.assert_array_equal(detector.transformer_sequences[7][-1],
                                              np.zeros(51, dtype=np.float32))


class AlarmConsumerTests(unittest.TestCase):
    def test_false_positive_recorder_requires_fresh_result_and_deduplicates(self):
        from experiments.analyze_false_positive import AlarmWindowRecorder
        recorder = AlarmWindowRecorder('normal.avi', 30, MagicMock(), before=0, after=0)
        with patch('builtins.print'):
            recorder.record_frame(30, {}, [7], [7])
            recorder.record_frame(31, {}, [7], [])
            recorder.record_frame(33, {}, [7], [7])
            recorder.record_frame(36, {}, [], [])
            recorder.record_frame(39, {}, [7], [7])
        self.assertEqual([event['alarm_frame'] for event in recorder.alarm_samples], [30, 39])

    def test_dashboard_only_records_fresh_inferences_and_respects_cooldown(self):
        ns = load_functions("dashboard/dashboard_app.py", {"update_statistics"},
                            {"time": MagicMock()}, class_name="FallDetectionDashboard")
        window = SimpleNamespace(
            stats={"total_people_detected": 0}, people_count_label=MagicMock(),
            fallen_timestamps={}, tracked_fallen_ids=set(), config={"fall_cooldown": 5.},
            _process_new_falls=MagicMock())
        data = {"person_count": 1, "fall_detected": True, "fallen_ids": [7],
                "fall_type": "transformer_fall", "fall_event_ids": [7]}
        ns["time"].time.return_value = 10
        ns["update_statistics"](window, data)
        window._process_new_falls.assert_called_once_with("transformer_fall", {7}, 10)
        # Even after cooldown elapses, a cached positive cannot create an event.
        ns["time"].time.return_value = 20
        ns["update_statistics"](window, {**data, "fall_event_ids": []})
        self.assertEqual(window._process_new_falls.call_count, 1)
        ns["update_statistics"](window, data)
        self.assertEqual(window._process_new_falls.call_count, 2)
        ns["time"].time.return_value = 21
        ns["update_statistics"](window, data)
        self.assertEqual(window._process_new_falls.call_count, 2)
        window.tracked_fallen_ids.add(7)
        ns["update_statistics"](window, {"person_count": 0, "expired_ids": [7]})
        self.assertNotIn(7, window.fallen_timestamps)
        self.assertNotIn(7, window.tracked_fallen_ids)

    def test_cli_does_not_record_cached_scores_or_keep_cleared_alarm_active(self):
        cv2, detector, utils = MagicMock(), MagicMock(), MagicMock()
        cap = cv2.VideoCapture.return_value
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, object())] * 6 + [(False, None)]
        cv2.waitKey.return_value = -1
        outputs = []
        for positive, fresh in [(True, False), (False, True), (True, True),
                                (True, False), (False, True), (True, True)]:
            data = {"person_ids": [7], "person_count": 1, "fallen_ids": [7] if positive else [],
                    "fall_event_ids": [7] if positive and fresh else [],
                    "fall_type": "transformer_fall" if positive else None}
            outputs.append((object(), positive, data))
        detector.process_frame.side_effect = outputs
        ns = load_functions("fall_detection_system.py", {"run_cli_mode"},
                            {"FallDetector": MagicMock(return_value=detector)})
        args = SimpleNamespace(model="unused.pt", conf=.5, source="unused.avi",
                               save_falls=True,
                               output_dir="unused")
        with patch.dict("sys.modules", {"cv2": cv2, "utils.utils": utils}):
            with patch("builtins.print"), patch("traceback.print_exc") as errors:
                ns["run_cli_mode"](args)
        errors.assert_not_called()
        self.assertEqual(utils.save_frame.call_count, 2)
        detector.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
