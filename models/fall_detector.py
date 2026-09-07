import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from ultralytics.engine.results import Boxes
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace
import math
import time
import torch
import os
import logging
from threading import RLock
from collections import defaultdict, deque
from utils.one_euro_filter import ONE_EURO_FILTER_PARAMS, OneEuroFilter
from models.transformer_fall_classifier import (
    FALL_CONFIDENCE_THRESHOLD,
    INPUT_TIMESTEPS,
    TRANSFORMER_INFERENCE_STRIDE,
    TransformerFallClassifier,
    extract_transformer_features,
)


class FallDetector:
    def __init__(self, model_path='yolov12n1.pt', confidence=0.5,
                 detector_confidence=0.3, transformer_model_path=None,
                 transformer_interpreter_class=None):
        """Initialize the fall detection system.
        
        Args:
            model_path (str): Path to the YOLOv12 model file (default: 'yolov12n1.pt')
            confidence (float): Confidence required for pose analysis and business logic
            detector_confidence (float): YOLO threshold for detections sent to ByteTrack
            transformer_model_path (str): Optional path to the 30-frame TFLite model
            transformer_interpreter_class: Optional compatible Interpreter class
        """
        self._lifecycle_lock = RLock()
        self._closed = False
        self.transformer_classifier = TransformerFallClassifier(
            model_path=transformer_model_path,
            interpreter_class=transformer_interpreter_class,
        )
        self.transformer_sequences = defaultdict(
            lambda: deque(maxlen=INPUT_TIMESTEPS)
        )
        self.transformer_probabilities = {}
        self.transformer_updates_since_inference = defaultdict(int)
        self.transformer_has_inferred = set()
        self.transformer_fall_threshold = FALL_CONFIDENCE_THRESHOLD
        self.transformer_inference_stride = TRANSFORMER_INFERENCE_STRIDE
        self._raw_pose_landmarks = {}
        self._expired_person_ids = []

        # This deployment runs on CPU only.
        try:
            print(f"PyTorch version: {torch.__version__}")
            self.device = "cpu"
            print("Using CPU for detection")
            
            # Initialize YOLOv12 model with explicit task
            print(f"Loading YOLOv12 model: {model_path}")
            
            # Download the model if needed, but DON'T overwrite model_path
            try:
                from ultralytics.utils.downloads import attempt_download
                downloaded_path = attempt_download(model_path)
                if downloaded_path:
                    print(f"Downloaded model to: {downloaded_path}")
                    model_path = str(downloaded_path)  # Only update if successful
                else:
                    print("Download returned None, using original path")
            except Exception as download_error:
                print(f"Model download error (continuing with original path): {download_error}")
            
            # Load model with explicit task type
            print(f"Using model path: {model_path}")
            self.model = YOLO(model_path, task='detect')
            self.model.to(self.device)
            print(f"Successfully loaded YOLOv12 model on {self.device}")
            
            # Verify model is on correct device
            model_device = next(self.model.parameters()).device
            print(f"Model confirmed on device: {model_device}")
            
            # Person class ID for COCO dataset
            self.person_class_id = 0
            self.detector_confidence = float(detector_confidence)
            self.analysis_confidence = float(confidence)
            
        except Exception as e:
            self.transformer_classifier.close()
            self.transformer_classifier = None
            print(f"Error during initialization: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Create MediaPipe graphs lazily, one per ByteTrack person ID.
        self.mp_pose = mp.solutions.pose
        self.pose_estimators = {}
        
        # Independent One Euro state and auxiliary pose history per tracked ID.
        self.pose_filter_params = ONE_EURO_FILTER_PARAMS.copy()
        self.pose_filters = {}
        self.pose_histories = defaultdict(lambda: deque(maxlen=15))
        self.fall_detected = False

        # Person tracking attributes
        tracker_config = YAML.load(ROOT / "cfg/trackers/bytetrack.yaml")
        tracker_config["track_low_thresh"] = self.detector_confidence
        tracker_config["track_high_thresh"] = max(self.analysis_confidence, self.detector_confidence)
        tracker_config["new_track_thresh"] = max(self.analysis_confidence, self.detector_confidence)
        self.tracker = BYTETracker(args=IterableSimpleNamespace(**tracker_config))
        self.person_states = {}  # Business state keyed by ByteTrack person ID
        self.fallen_person_ids = set()  # Set of IDs of persons who are currently fallen

    def _close_pose_estimator(self, person_id):
        """Close under the lifecycle lock; retain failed graphs for a later retry."""
        estimator = self.pose_estimators.get(person_id)
        if estimator is not None:
            try:
                estimator.close()
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to close Pose for person %s", person_id
                )
            else:
                del self.pose_estimators[person_id]

    def _person_state_mappings(self):
        """All per-person dictionaries except Pose, whose close may need retry."""
        return (
            self.person_states, self.pose_filters, self.pose_histories,
            self.transformer_sequences, self.transformer_probabilities,
            self.transformer_updates_since_inference, self._raw_pose_landmarks,
        )

    def _clear_person_state(self, person_id):
        """Release one person under the caller's lifecycle lock."""
        self._close_pose_estimator(person_id)
        for mapping in self._person_state_mappings():
            mapping.pop(person_id, None)
        self.fallen_person_ids.discard(person_id)
        self.transformer_has_inferred.discard(person_id)

    def close(self):
        """Wait for active processing, then release Pose graphs. Safe to repeat.

        A closed detector cannot restart processing; create a new detector instead.
        """
        with self._lifecycle_lock:
            self._closed = True
            person_ids = set(self.pose_estimators)
            for mapping in self._person_state_mappings():
                person_ids.update(mapping)
            person_ids.update(self.fallen_person_ids)
            person_ids.update(self.transformer_has_inferred)
            for person_id in person_ids:
                self._clear_person_state(person_id)
            self._expired_person_ids.clear()
            self.fall_detected = False
            if self.transformer_classifier is not None:
                self.transformer_classifier.close()
                self.transformer_classifier = None

    @property
    def confidence(self):
        """Backward-compatible alias for the pose/analysis confidence threshold."""
        return self.analysis_confidence

    @confidence.setter
    def confidence(self, value):
        """Update the business threshold without changing the YOLO detector threshold."""
        self.analysis_confidence = float(value)
        if hasattr(self, "tracker"):
            tracker_threshold = max(self.analysis_confidence, self.detector_confidence)
            self.tracker.args.track_high_thresh = tracker_threshold
            self.tracker.args.new_track_thresh = tracker_threshold

    def detect_person(self, frame):
        """Detect persons with YOLO and return bbox, confidence and class data."""
        # Process frame with explicit device
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype != 'uint8':
                    frame = cv2.convertScaleAbs(frame)
                
            # Run inference with explicit device
            results = self.model(frame, 
                                verbose=False, 
                                conf=self.detector_confidence,
                                device=self.device)
            
            person_detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    if cls == self.person_class_id and conf >= self.detector_confidence:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        person_detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": float(conf),
                            "class_id": cls
                        })
            
            return person_detections
        except Exception as e:
            print(f"Error in detect_person: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def analyze_pose(self, frame, person_box, person_id):
        """Analyze a valid tracked person without racing graph cleanup."""
        with self._lifecycle_lock:
            if (self._closed or isinstance(person_id, (bool, np.bool_))
                    or not isinstance(person_id, (int, np.integer)) or person_id <= 0):
                return None, None
            self._raw_pose_landmarks.pop(person_id, None)
            actual_box = self._clip_person_box(frame, person_box)
            if actual_box is None:
                return None, None
            return self._analyze_pose(frame, actual_box, int(person_id))

    @staticmethod
    def _clip_person_box(frame, person_box):
        """Return integer bounds constrained to the frame, or None if empty."""
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = (int(value) for value in person_box)
        x1 = max(0, min(frame_width, x1))
        y1 = max(0, min(frame_height, y1))
        x2 = max(0, min(frame_width, x2))
        y2 = max(0, min(frame_height, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _analyze_pose(self, frame, person_box, person_id):
        """Analyze pose for a detected person using MediaPipe.
        
        Args:
            frame: Input frame
            person_box: Person bounding box (x1, y1, x2, y2)
            person_id: Tracking ID used to select this person's pose history
            
        Returns:
            tuple: (landmarks, pose_features) or (None, None) if pose detection fails
        """
        x1, y1, x2, y2 = person_box
        
        # Extract the person from the frame
        person_img = frame[y1:y2, x1:x2]
        
        if person_img.size == 0:
            return None, None
        
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        
        # Reuse this person's temporal MediaPipe graph across frames.
        estimator = self.pose_estimators.get(person_id)
        if estimator is None:
            estimator = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose_estimators[person_id] = estimator
        results = estimator.process(image=rgb_img)
        
        if not results.pose_landmarks:
            return None, None
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))

        # Keep an independent, immutable raw copy before One Euro smoothing.
        self._raw_pose_landmarks[person_id] = tuple(landmarks)

        # Smooth every landmark before computing or storing temporal features.
        landmarks = self.smooth_pose_landmarks(landmarks, person_id, time.monotonic())

        # Calculate pose features
        pose_features = self.calculate_pose_features(landmarks, person_id)
        
        return landmarks, pose_features
    


    def smooth_pose_landmarks(self, landmarks, person_id, timestamp):
        """Smooth all 33 pose landmarks with per-person, per-axis filter state."""
        person_filters = self.pose_filters.setdefault(person_id, {})
        smoothed_landmarks = []
        for landmark_id, (x, y, z, visibility) in enumerate(landmarks):
            if landmark_id not in person_filters:
                person_filters[landmark_id] = {
                    axis: OneEuroFilter(**self.pose_filter_params)
                    for axis in ("x", "y", "z")
                }
            filters = person_filters[landmark_id]
            filtered_coordinates = (
                filters["x"](x, timestamp),
                filters["y"](y, timestamp),
                filters["z"](z, timestamp),
            )
            smoothed_landmarks.append((*filtered_coordinates, visibility))
        return smoothed_landmarks

    def calculate_pose_features(self, landmarks, person_id):
        """Calculate pose features from landmarks.
        
        Args:
            landmarks: List of pose landmarks
            person_id: Tracking ID used to select this person's pose history
            
        Returns:
            dict: Dictionary of pose features or None if calculation fails
        """
        if not landmarks:
            return None
        
        # Extract key landmarks (indices may vary based on MediaPipe version)
        # Head, shoulders, hips, knees, ankles
        key_points = [0, 11, 12, 23, 24, 25, 26, 27, 28]
        
        # Extract x, y coordinates of key points
        key_landmarks = [landmarks[i][:2] for i in key_points if i < len(landmarks)]
        
        if len(key_landmarks) < len(key_points):
            return None
        
        # Calculate height of the person (vertical distance between head and feet)
        head_y = landmarks[0][1]
        left_ankle_y = landmarks[27][1]
        right_ankle_y = landmarks[28][1]
        ankle_y = (left_ankle_y + right_ankle_y) / 2
        height = abs(ankle_y - head_y)
        
        # Calculate orientation (angle with the vertical)
        # Using the spine (mid-point of shoulders to mid-point of hips)
        mid_shoulder_x = (landmarks[11][0] + landmarks[12][0]) / 2
        mid_shoulder_y = (landmarks[11][1] + landmarks[12][1]) / 2
        mid_hip_x = (landmarks[23][0] + landmarks[24][0]) / 2
        mid_hip_y = (landmarks[23][1] + landmarks[24][1]) / 2
        
        dx = mid_hip_x - mid_shoulder_x
        dy = mid_hip_y - mid_shoulder_y
        
        angle = math.degrees(math.atan2(dx, dy))  # Angle with vertical axis
        
        # Retain body positions for auxiliary posture and false-positive analysis
        shoulder_pos = (mid_shoulder_x, mid_shoulder_y)
        hip_pos = (mid_hip_x, mid_hip_y)
        feet_pos = ((landmarks[27][0] + landmarks[28][0])/2, (landmarks[27][1] + landmarks[28][1])/2)
        
        # Calculate bounding box aspect ratio (height/width) - helps detect lying down
        left_most_x = min(landmarks[11][0], landmarks[23][0], landmarks[25][0], landmarks[27][0])
        right_most_x = max(landmarks[12][0], landmarks[24][0], landmarks[26][0], landmarks[28][0])
        top_most_y = min(landmarks[0][1], landmarks[11][1], landmarks[12][1])
        bottom_most_y = max(landmarks[27][1], landmarks[28][1])
        
        bbox_width = right_most_x - left_most_x
        bbox_height = bottom_most_y - top_most_y
        aspect_ratio = bbox_height / max(bbox_width, 0.0001)  # Avoid division by zero
        
        # Calculate distances between key points
        shoulder_to_hip_distance = math.sqrt((mid_shoulder_x - mid_hip_x)**2 + (mid_shoulder_y - mid_hip_y)**2)
        hip_to_feet_distance = math.sqrt((mid_hip_x - feet_pos[0])**2 + (mid_hip_y - feet_pos[1])**2)
        
        # Calculate motion features using only this person's pose history.
        pose_history = self.pose_histories[person_id]
        velocity_y = 0  # Vertical velocity
        velocity_x = 0  # Horizontal velocity
        acceleration = 0
        jerk = 0  # Rate of change of acceleration
        
        if pose_history:
            prev = pose_history[-1]
            time_diff = 1  # Assuming consistent frame rate for simplicity
            
            # Calculate vertical velocity
            velocity_y = (mid_shoulder_y - prev["mid_shoulder_y"]) / time_diff
            
            # Calculate horizontal velocity
            velocity_x = (mid_shoulder_x - prev["mid_shoulder_x"]) / time_diff
            
            # Calculate acceleration if we have at least 2 previous poses
            if len(pose_history) >= 2:
                prev_velocity_y = (prev["mid_shoulder_y"] - pose_history[-2]["mid_shoulder_y"]) / time_diff
                acceleration = (velocity_y - prev_velocity_y) / time_diff
                
                # Calculate jerk if we have at least 3 previous poses
                if len(pose_history) >= 3:
                    prev_acceleration = (prev_velocity_y - (pose_history[-2]["mid_shoulder_y"] - pose_history[-3]["mid_shoulder_y"]) / time_diff) / time_diff
                    jerk = (acceleration - prev_acceleration) / time_diff
        
        features = {
            "height": height,
            "angle": angle,
            "velocity_y": velocity_y,
            "velocity_x": velocity_x,
            "acceleration": acceleration,
            "jerk": jerk,
            "mid_shoulder_y": mid_shoulder_y,
            "mid_shoulder_x": mid_shoulder_x,
            "mid_hip_y": mid_hip_y,
            "mid_hip_x": mid_hip_x,
            "shoulder_pos": shoulder_pos,
            "hip_pos": hip_pos,
            "feet_pos": feet_pos,
            "aspect_ratio": aspect_ratio,
            "shoulder_to_hip_distance": shoulder_to_hip_distance,
            "hip_to_feet_distance": hip_to_feet_distance,
            "timestamp": time.time()
        }
        
        # deque(maxlen=15) automatically discards the oldest pose.
        pose_history.append(features)
        
        return features
    
    
    
    def process_frame(self, frame):
        """Process a frame atomically with respect to detector shutdown."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("FallDetector is closed")
            return self._process_frame(frame)

    def _update_transformer_sequence(self, person_id, features):
        """Append once per pose attempt; infer at updates 30, 33, 36, ... ."""
        sequence = self.transformer_sequences[person_id]
        sequence.append(features)
        self.transformer_updates_since_inference[person_id] += 1
        inferred = False
        if len(sequence) == INPUT_TIMESTEPS and (
            person_id not in self.transformer_has_inferred
            or self.transformer_updates_since_inference[person_id]
            >= self.transformer_inference_stride
        ):
            probability = self.transformer_classifier.predict(sequence)
            self.transformer_probabilities[person_id] = probability
            self.transformer_has_inferred.add(person_id)
            self.transformer_updates_since_inference[person_id] = 0
            inferred = True
        return self.transformer_probabilities.get(person_id), inferred

    def _process_frame(self, frame):
        """Process a single frame for fall detection.
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (output_frame, fall_detected, fall_data)
        """
        # Make a copy to avoid modifying the original
        output_frame = frame.copy()
        
        # Detect persons in the frame, then assign persistent ByteTrack IDs.
        person_detections = self.detect_person(frame)
        person_boxes, box_to_id = self.update_tracker(frame, person_detections)
        
        # List to store all person IDs in the current frame
        current_person_ids = []
        
        falls_detected = False
        fall_data = {
            "person_count": 0,
            "fall_detected": False,
            "fall_type": None,
            "person_boxes": [],
            "critical_points": [],
            "person_ids": current_person_ids,
            "fallen_ids": [],
            "fall_event_ids": [],
            "transformer_probabilities": {},
            "transformer_sequence_lengths": {},
            "transformer_has_inferred": {},
            "inferred_this_frame": {},
            "expired_ids": list(self._expired_person_ids),
        }
        
        # Store landmarks and features for each person
        pose_data = []
        
        # Process each detected person to get pose data (we'll use this for our enhanced detection)
        for i, box in enumerate(person_boxes):
            person_id = box_to_id.get(i)
            if (isinstance(person_id, (bool, np.bool_))
                    or not isinstance(person_id, (int, np.integer)) or person_id <= 0):
                continue
            actual_box = self._clip_person_box(frame, box)
            if actual_box is None:
                self._raw_pose_landmarks.pop(person_id, None)
                continue
            display_box = actual_box
            current_person_ids.append(person_id)
            fall_data["person_boxes"].append(display_box)

            # analyze_pose uses these same clipped bounds for its crop.
            self._raw_pose_landmarks.pop(person_id, None)
            landmarks, pose_features = self.analyze_pose(frame, display_box, person_id)
            transformer_features = extract_transformer_features(
                self._raw_pose_landmarks.get(person_id), display_box, frame.shape
            )
            probability, inferred = self._update_transformer_sequence(
                person_id, transformer_features
            )

            fall_data["transformer_probabilities"][person_id] = probability
            fall_data["transformer_sequence_lengths"][person_id] = len(
                self.transformer_sequences[person_id]
            )
            fall_data["transformer_has_inferred"][person_id] = (
                person_id in self.transformer_has_inferred
            )
            fall_data["inferred_this_frame"][person_id] = inferred
            pose_data.append(
                (person_id, landmarks, pose_features, display_box, probability)
            )

        # The Transformer probability is the sole final fall decision.
        for person_id, landmarks, pose_features, box, probability in pose_data:
            x1, y1, x2, y2 = box
            is_fall = (
                probability is not None
                and probability >= self.transformer_fall_threshold
            )
            person_state = self.person_states.get(person_id)
            if is_fall:
                fall_data["fallen_ids"].append(person_id)
                if fall_data["inferred_this_frame"][person_id]:
                    fall_data["fall_event_ids"].append(person_id)
                self.fallen_person_ids.add(person_id)
                if person_state is not None:
                    person_state["is_fallen"] = True
                falls_detected = True
            else:
                self.fallen_person_ids.discard(person_id)
                if person_state is not None:
                    person_state["is_fallen"] = False

            if probability is None:
                status_text = (
                    "Collecting pose sequence: "
                    f"{fall_data['transformer_sequence_lengths'][person_id]}/{INPUT_TIMESTEPS}"
                )
                status_color = (0, 255, 255)
            else:
                status_text = f"Fall probability: {probability:.3f}"
                status_color = (0, 0, 255) if is_fall else (0, 255, 0)
            cv2.putText(
                output_frame, status_text, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2
            )

            if is_fall:
                cv2.putText(
                    output_frame, "FALL DETECTED: transformer_fall",
                    (x1, max(45, y1 - 35)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2
                )

        fall_data["fall_detected"] = falls_detected
        fall_data["person_count"] = len(current_person_ids)
        if falls_detected:
            fall_data["fall_type"] = "transformer_fall"
        self.fall_detected = falls_detected
        
        return output_frame, falls_detected, fall_data
    
    def update_tracker(self, frame, person_detections):
        """Update tracks and expire Pose graphs without racing pose analysis."""
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("FallDetector is closed")
            return self._update_tracker(frame, person_detections)

    def _update_tracker(self, frame, person_detections):
        """Update ByteTrack with YOLO detections and return tracked boxes and IDs.

        Args:
            frame: Input frame used by the tracker.
            person_detections: YOLO person detections containing bbox, confidence and class_id.

        Returns:
            tuple: (person_boxes, box_to_id), where box_to_id maps each returned
            box index to its ByteTrack person ID.
        """
        if person_detections:
            detection_data = np.asarray([
                [*detection["bbox"], detection["confidence"], detection["class_id"]]
                for detection in person_detections
            ], dtype=np.float32)
        else:
            detection_data = np.empty((0, 6), dtype=np.float32)

        detections = Boxes(detection_data, frame.shape[:2])
        tracks = self.tracker.update(detections, frame)

        current_time = time.time()
        person_boxes = []
        box_to_id = {}

        # BYTETracker rows are [x1, y1, x2, y2, track_id, score, class_id, detection_index].
        for track in tracks:
            class_id = int(track[6])
            if class_id != self.person_class_id:
                continue

            x1, y1, x2, y2 = np.asarray(track[:4]).astype(int)
            person_id = int(track[4])
            track_confidence = float(track[5])

            state = self.person_states.setdefault(person_id, {
                "last_seen": current_time,
                "is_fallen": False
            })
            state["last_seen"] = current_time
            state["is_fallen"] = person_id in self.fallen_person_ids

            # Low-confidence detections can maintain ByteTrack state, but only
            # sufficiently confident active tracks proceed to MediaPipe Pose.
            if track_confidence < self.analysis_confidence:
                continue

            box_index = len(person_boxes)
            person_boxes.append((x1, y1, x2, y2))
            box_to_id[box_index] = person_id

        # Remove business state and pose history for IDs absent for over 5 seconds.
        expired_ids = [
            person_id
            for person_id, state in self.person_states.items()
            if current_time - state["last_seen"] > 5.0
        ]
        self._expired_person_ids = expired_ids
        for person_id in expired_ids:
            self._clear_person_state(person_id)

        return person_boxes, box_to_id
