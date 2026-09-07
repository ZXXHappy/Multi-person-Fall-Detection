"""Feature preparation and TFLite inference for the 30-frame fall model.

Normalization intentionally mirrors deployment/raspberry_pi/fall-detector.py
from punpayut/Fall-Detection. The published model uses alphabetically sorted
keypoint names, represented by TRANSFORMER_KEYPOINT_INDICES below.
"""
from importlib import import_module
from pathlib import Path

import numpy as np


INPUT_TIMESTEPS = 30
TRANSFORMER_INFERENCE_STRIDE = 3
NUM_FEATURES = 51
FALL_CONFIDENCE_THRESHOLD = 0.9
MIN_NORMALIZATION_VISIBILITY = 0.3

# Alphabetical keypoint-name order used by the published deployment script:
# Left Ankle, Left Ear, Left Elbow, Left Eye, Left Hip, Left Knee,
# Left Shoulder, Left Wrist, Nose, then the corresponding Right keypoints.
TRANSFORMER_KEYPOINT_INDICES = (
    27, 7, 13, 2, 23, 25, 11, 15, 0,
    28, 8, 14, 5, 24, 26, 12, 16,
)
TRANSFORMER_KEYPOINT_NAMES = (
    "Left Ankle", "Left Ear", "Left Elbow", "Left Eye", "Left Hip",
    "Left Knee", "Left Shoulder", "Left Wrist", "Nose", "Right Ankle",
    "Right Ear", "Right Elbow", "Right Eye", "Right Hip", "Right Knee",
    "Right Shoulder", "Right Wrist",
)
KEYPOINT_DICT = {name: index for index, name in enumerate(TRANSFORMER_KEYPOINT_NAMES)}


class TFLiteUnavailableError(RuntimeError):
    """Raised when none of the supported TFLite interpreter packages exists."""


def _feature_indices(keypoint_name):
    keypoint_index = KEYPOINT_DICT[keypoint_name]
    start = keypoint_index * 3
    return start, start + 1, start + 2


def normalize_skeleton_frame(
        frame_features, min_confidence=MIN_NORMALIZATION_VISIBILITY):
    """Apply the exact normalization used by the model's deployment script."""
    frame_features = np.asarray(frame_features, dtype=np.float32)
    if frame_features.shape != (NUM_FEATURES,):
        raise ValueError(
            f"Expected ({NUM_FEATURES},) skeleton features, got {frame_features.shape}"
        )
    normalized_frame = np.copy(frame_features)

    ls_x_idx, ls_y_idx, ls_c_idx = _feature_indices("Left Shoulder")
    rs_x_idx, rs_y_idx, rs_c_idx = _feature_indices("Right Shoulder")
    lh_x_idx, lh_y_idx, lh_c_idx = _feature_indices("Left Hip")
    rh_x_idx, rh_y_idx, rh_c_idx = _feature_indices("Right Hip")

    ls_x, ls_y, ls_c = frame_features[[ls_x_idx, ls_y_idx, ls_c_idx]]
    rs_x, rs_y, rs_c = frame_features[[rs_x_idx, rs_y_idx, rs_c_idx]]
    lh_x, lh_y, lh_c = frame_features[[lh_x_idx, lh_y_idx, lh_c_idx]]
    rh_x, rh_y, rh_c = frame_features[[rh_x_idx, rh_y_idx, rh_c_idx]]

    mid_shoulder_x, mid_shoulder_y = np.nan, np.nan
    valid_ls, valid_rs = ls_c > min_confidence, rs_c > min_confidence
    if valid_ls and valid_rs:
        mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
    elif valid_ls:
        mid_shoulder_x, mid_shoulder_y = ls_x, ls_y
    elif valid_rs:
        mid_shoulder_x, mid_shoulder_y = rs_x, rs_y

    mid_hip_x, mid_hip_y = np.nan, np.nan
    valid_lh, valid_rh = lh_c > min_confidence, rh_c > min_confidence
    if valid_lh and valid_rh:
        mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
    elif valid_lh:
        mid_hip_x, mid_hip_y = lh_x, lh_y
    elif valid_rh:
        mid_hip_x, mid_hip_y = rh_x, rh_y

    if np.isnan(mid_hip_x) or np.isnan(mid_hip_y):
        return np.nan_to_num(frame_features, nan=0.0, posinf=0.0, neginf=0.0)

    reference_height = np.nan
    if not np.isnan(mid_shoulder_y) and not np.isnan(mid_hip_y):
        reference_height = np.abs(mid_shoulder_y - mid_hip_y)
    perform_scaling = not (np.isnan(reference_height) or reference_height < 1e-5)

    for keypoint_name in TRANSFORMER_KEYPOINT_NAMES:
        x_index, y_index, _ = _feature_indices(keypoint_name)
        normalized_frame[x_index] -= mid_hip_x
        normalized_frame[y_index] -= mid_hip_y
        if perform_scaling:
            normalized_frame[x_index] /= reference_height
            normalized_frame[y_index] /= reference_height
    return np.nan_to_num(normalized_frame, nan=0.0, posinf=0.0, neginf=0.0)


def extract_transformer_features(landmarks, person_box, frame_shape):
    """Restore crop-relative landmarks to full-frame coordinates and normalize."""
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    if not landmarks:
        return features

    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = person_box
    x1 = max(0, min(frame_width, int(x1)))
    y1 = max(0, min(frame_height, int(y1)))
    x2 = max(0, min(frame_width, int(x2)))
    y2 = max(0, min(frame_height, int(y2)))
    crop_width, crop_height = x2 - x1, y2 - y1
    if frame_width <= 0 or frame_height <= 0 or crop_width <= 0 or crop_height <= 0:
        return features

    for feature_index, landmark_index in enumerate(TRANSFORMER_KEYPOINT_INDICES):
        if landmark_index >= len(landmarks):
            continue
        x_crop, y_crop, _, visibility = landmarks[landmark_index]
        start = feature_index * 3
        features[start] = (x1 + x_crop * crop_width) / frame_width
        features[start + 1] = (y1 + y_crop * crop_height) / frame_height
        features[start + 2] = visibility
    return normalize_skeleton_frame(features)


INTERPRETER_MODULES = (
    "ai_edge_litert.interpreter",
    "tflite_runtime.interpreter",
    "tensorflow",
)


class TransformerFallClassifier:
    """Validated wrapper around the published float32 TFLite model."""

    def __init__(self, model_path=None, interpreter_class=None):
        if model_path is None:
            model_path = Path(__file__).resolve().with_name("fall_detection_transformer.tflite")
        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parent / self.model_path
        self.model_path = self.model_path.resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Transformer model not found: {self.model_path}")

        self.interpreter = None
        self.input_detail = self.output_detail = None
        self.backend = None
        self.load_errors = []
        if interpreter_class is not None:
            try:
                self._initialize_interpreter(interpreter_class)
                self.backend = interpreter_class.__name__
            except Exception:
                self.close()
                raise
            return

        # Importing is not enough: allocation and tensor contract validation
        # must also succeed before this backend is selected.
        for module_name in INTERPRETER_MODULES:
            try:
                module = import_module(module_name)
                candidate = (module.lite.Interpreter if module_name == "tensorflow"
                             else module.Interpreter)
                self._initialize_interpreter(candidate)
                self.backend = module_name
                return
            except Exception as exc:
                self.load_errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
                self.close()
        raise TFLiteUnavailableError(
            "No available TFLite interpreter could load the model with the required "
            "float32 (1,30,51) -> (1,1) interface. No rule-based fallback is enabled. "
            "Checked ai-edge-litert, tflite_runtime, tensorflow. "
            + " | ".join(self.load_errors)
        )

    def _initialize_interpreter(self, interpreter_class):
        # Python handles Unicode paths on Windows; some LiteRT native file
        # loaders cannot open them. Loading the same bytes avoids that issue
        # without depending on the process working directory.
        self.interpreter = interpreter_class(model_content=self.model_path.read_bytes())
        self.interpreter.allocate_tensors()
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        if len(input_details) != 1 or len(output_details) != 1:
            raise ValueError("Transformer model must have exactly one input and one output")
        self.input_detail = input_details[0]
        self.output_detail = output_details[0]

        input_shape = tuple(int(value) for value in self.input_detail["shape"])
        output_shape = tuple(int(value) for value in self.output_detail["shape"])
        if input_shape != (1, INPUT_TIMESTEPS, NUM_FEATURES):
            raise ValueError(f"Unexpected Transformer input shape: {input_shape}")
        if np.dtype(self.input_detail["dtype"]) != np.dtype(np.float32):
            raise ValueError(f"Unexpected Transformer input dtype: {self.input_detail['dtype']}")
        if output_shape != (1, 1):
            raise ValueError(f"Unexpected Transformer output shape: {output_shape}")
        if np.dtype(self.output_detail["dtype"]) != np.dtype(np.float32):
            raise ValueError(f"Unexpected Transformer output dtype: {self.output_detail['dtype']}")

    def predict(self, sequence):
        if self.interpreter is None:
            raise RuntimeError("TransformerFallClassifier is closed")
        model_input = np.asarray(sequence, dtype=np.float32)
        if model_input.shape != (INPUT_TIMESTEPS, NUM_FEATURES):
            raise ValueError(
                "Transformer sequence must have shape "
                f"({INPUT_TIMESTEPS}, {NUM_FEATURES}), got {model_input.shape}"
            )
        model_input = np.expand_dims(model_input, axis=0)
        self.interpreter.set_tensor(self.input_detail["index"], model_input)
        self.interpreter.invoke()
        output = np.asarray(self.interpreter.get_tensor(self.output_detail["index"]))
        if output.shape != (1, 1) or output.dtype != np.float32:
            raise RuntimeError(f"Unexpected Transformer output: {output.shape}, {output.dtype}")
        probability = float(output[0, 0])
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Invalid Transformer Fall probability: {probability}")
        return probability

    def close(self):
        """Drop references; Python TFLite APIs expose no explicit close method."""
        self.input_detail = None
        self.output_detail = None
        self.interpreter = None
