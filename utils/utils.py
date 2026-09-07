import cv2
import numpy as np
import os
import datetime
import json
import math


def create_logger(log_file_path='logs/fall_detection_log.json'):
    """Create a logger that saves detection events to a JSON file.
    
    Args:
        log_file_path: Path to the log file
    
    Returns:
        A logger function that can be called to log events
    """
    # Make sure the directory exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Initialize the log file if it doesn't exist
    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as f:
            json.dump([], f)
    
    def log_event(event_type, event_data):
        """Log an event to the JSON file.
        
        Args:
            event_type: Type of event (e.g., 'fall_detected', 'system_start')
            event_data: Dictionary of event data
        """
        # Read existing logs
        try:
            with open(log_file_path, 'r') as f:
                logs = json.load(f)
        except:
            logs = []
        
        # Add timestamp to event data
        event_data['timestamp'] = datetime.datetime.now().isoformat()
        event_data['event_type'] = event_type
        
        # Append new log
        logs.append(event_data)
        
        # Write updated logs back to file
        with open(log_file_path, 'w') as f:
            json.dump(logs, f, indent=2)
    
    return log_event


def draw_skeleton(frame, landmarks, connections, box=None):
    """Draw a skeleton on the frame based on landmarks.
    
    Args:
        frame: The frame to draw on
        landmarks: List of (x, y) normalized coordinates
        connections: List of tuples (start_idx, end_idx) for skeleton connections
        box: Optional bounding box (x1, y1, x2, y2) for scaling coordinates
        
    Returns:
        Frame with skeleton drawn
    """
    h, w = frame.shape[:2]
    
    if box:
        x1, y1, x2, y2 = box
        box_w, box_h = x2 - x1, y2 - y1
        
        # Convert normalized coordinates to pixel coordinates within the box
        pixel_landmarks = []
        for lm in landmarks:
            x, y = int(lm[0] * box_w), int(lm[1] * box_h)
            pixel_landmarks.append((x + x1, y + y1))
    else:
        # Convert normalized coordinates to pixel coordinates for the whole frame
        pixel_landmarks = []
        for lm in landmarks:
            x, y = int(lm[0] * w), int(lm[1] * h)
            pixel_landmarks.append((x, y))
    
    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        if (start_idx < len(pixel_landmarks) and end_idx < len(pixel_landmarks)):
            cv2.line(
                frame,
                pixel_landmarks[start_idx],
                pixel_landmarks[end_idx],
                (245, 66, 230),
                2
            )
    
    # Draw landmark points
    for point in pixel_landmarks:
        cv2.circle(frame, point, 4, (245, 117, 66), -1)
    
    return frame


def calculate_angle(point1, point2, point3):
    """Calculate the angle formed by three points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (vertex of the angle) (x, y)
        point3: Third point (x, y)
    
    Returns:
        Angle in degrees
    """
    # Vectors from point2 to point1 and point2 to point3
    a = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    b = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
    # Dot product
    dot_product = np.dot(a, b)
    
    # Magnitudes
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    
    # Angle
    cos_angle = dot_product / (mag_a * mag_b)
    
    # Ensure cos_angle is within -1 to 1 range due to potential floating point issues
    cos_angle = max(-1, min(1, cos_angle))
    
    angle = math.degrees(math.acos(cos_angle))
    return angle


def save_frame(frame, output_dir='fall_snapshots', prefix='fall'):
    """Save a frame as an image file.
    
    Args:
        frame: The frame to save
        output_dir: Directory to save the image in
        prefix: Prefix for the filename
        
    Returns:
        Path to the saved image
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = os.path.join(output_dir, filename)
    
    # Save the image
    cv2.imwrite(filepath, frame)
    
    return filepath


def get_euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance between the points
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def is_valid_pose(landmarks):
    """Check if a pose is valid based on basic sanity checks.
    
    Args:
        landmarks: List of pose landmarks
        
    Returns:
        Boolean indicating if the pose is valid
    """
    # Check if we have enough landmarks
    if len(landmarks) < 33:  # MediaPipe has 33 pose landmarks
        return False
    
    # Check if key landmarks have reasonable visibility
    key_indices = [0, 11, 12, 23, 24, 27, 28]  # nose, shoulders, hips, ankles
    for idx in key_indices:
        if idx < len(landmarks) and landmarks[idx][3] < 0.5:  # Visibility check
            return False
    
    return True


def detect_overlapping_people(person_boxes, iou_threshold=0.3):
    """Detect overlapping person bounding boxes that might cause confusion.
    
    Args:
        person_boxes: List of person bounding boxes [(x1, y1, x2, y2), ...]
        iou_threshold: Threshold for considering boxes as overlapping
        
    Returns:
        List of indices of boxes that should be processed with caution
    """
    caution_indices = []
    
    for i in range(len(person_boxes)):
        for j in range(i+1, len(person_boxes)):
            box1 = person_boxes[i]
            box2 = person_boxes[j]
            
            # Calculate IoU
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 < x1 or y2 < y1:
                # No overlap
                continue
                
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            iou = intersection / union
            
            if iou > iou_threshold:
                if i not in caution_indices:
                    caution_indices.append(i)
                if j not in caution_indices:
                    caution_indices.append(j)
    
    return caution_indices 