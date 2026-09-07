#!/usr/bin/env python3
import os
import sys
import argparse
from models.fall_detector import FallDetector
from dashboard.dashboard_app import FallDetectionDashboard
import tkinter as tk

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fall Detection System')
    
    # Main options
    parser.add_argument('--mode', type=str, default='dashboard', choices=['dashboard', 'cli'],
                      help='Application mode: dashboard (GUI) or cli (command line)')
    
    # Model options
    parser.add_argument('--model', type=str, default='yolov12n1.pt',
                      help='Path to YOLOv12 model file (default: yolov12n1.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                      help='Detection confidence threshold (0-1)')
    
    # Video source options
    parser.add_argument('--source', type=str, default='0',
                      help='Video source (0 for webcam, or path to video file)')
    
    # Output options
    parser.add_argument('--save-falls', action='store_true',
                      help='Save frames when falls are detected')
    parser.add_argument('--output-dir', type=str, default='fall_snapshots',
                      help='Directory to save fall snapshots (if --save-falls is used)')
    
    return parser.parse_args()

def run_dashboard_mode(args):
    """Run the fall detection system in dashboard mode."""
    try:
        from PyQt5.QtWidgets import QApplication
        from dashboard.dashboard_app import FallDetectionDashboard
        
        app = QApplication(sys.argv)
        window = FallDetectionDashboard()
        try:
            window.show()
            sys.exit(app.exec_())
        finally:
            window.stop_detection()
    except ImportError as e:
        print(f"Error: Required PyQt5 libraries not found. {str(e)}")
        print("Please install PyQt5 using: pip install PyQt5")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting dashboard: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def run_cli_mode(args):
    """Run the system in command line mode."""
    import cv2
    import time
    from utils.utils import save_frame
    
    print("Starting Fall Detection System in CLI mode")
    print(f"Using model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    
    # Initialize fall detector
    fall_detector = FallDetector(
        model_path=args.model,
        confidence=args.conf
    )
    
    cap = None
    try:
        # Open video source
        if args.source.isdigit():
            source_id = int(args.source)
            print(f"Opening webcam with ID: {source_id}")
            cap = cv2.VideoCapture(source_id, cv2.CAP_DSHOW)  # Try DSHOW backend on Windows
        else:
            print(f"Opening video file: {args.source}")
            cap = cv2.VideoCapture(args.source)
    
        if not cap.isOpened():
            print(f"Error: Could not open video source {args.source}")
            return
    
        # Print camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video source opened with resolution: {width}x{height}, FPS: {fps}")
    
        # Initialize FPS calculation
        frame_count = 0
        start_time = time.time()
        last_fps_update = start_time
    
        # Add tracking for fallen people and total fall count
        active_falls = set()  # Track currently fallen people by ID
        total_falls = 0       # Count of total unique falls
    
        print("Press 'q' to quit, 's' to save current frame")
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to receive frame. Exiting...")
                break
            
            frame_count += 1
            
            # Process the frame
            output_frame, fall_detected, fall_data = fall_detector.process_frame(frame)
            
            # Update fall tracking
            if 'person_ids' in fall_data and 'fallen_ids' in fall_data:
                # Get people who are currently fallen
                currently_fallen = set(fall_data['fallen_ids'])
                
                # Find newly fallen people (not already in active_falls)
                new_falls = set(fall_data.get('fall_event_ids', [])) - active_falls
                if new_falls:
                    print("\nFALL DETECTED: transformer_fall")
                    # Increment total falls count by number of new falls
                    total_falls += len(new_falls)
                    
                    # If saving falls is enabled, save a frame for each new fall
                    if args.save_falls:
                        fall_type = fall_data["fall_type"] or "unknown_type"
                        saved_path = save_frame(output_frame, args.output_dir, f"fall_{fall_type}")
                        print(f"\nFall snapshot saved to: {saved_path}")
                
                # Update our tracking of active falls
                active_falls = currently_fallen
                
                # Remove IDs of people who are no longer in the frame
                if 'person_ids' in fall_data:
                    all_people = set(fall_data['person_ids'])
                    active_falls &= all_people  # Keep only IDs still present
            
            # Display statistics and status
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Update FPS every second
            if current_time - last_fps_update >= 1.0:
                current_fps = frame_count / elapsed_time
                last_fps_update = current_time
                
                # Display basic stats
                print(f"\rFPS: {current_fps:.1f} | People: {fall_data['person_count']} | Current falls: {len(active_falls)} | Total falls: {total_falls}", end='')
                
            
            # Display the output frame
            cv2.imshow('Fall Detection', output_frame)
            
            # Handle keyboard commands
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                saved_path = save_frame(output_frame, args.output_dir, "manual_save")
                print(f"\nFrame saved to: {saved_path}")
    
    except KeyboardInterrupt:
        print("\nDetection stopped by user")
    except Exception as e:
        print(f"\nError during detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release resources
        print("\nReleasing resources...")
        fall_detector.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Check if the YOLOv12 model file exists
    if not os.path.exists(args.model) and not args.model.startswith('yolov12'):
        print(f"Warning: Model file '{args.model}' not found.")
        print("The system will attempt to download it if it's a standard YOLOv12 model.")
    
    # Run in the selected mode
    if args.mode == 'dashboard':
        run_dashboard_mode(args)
    else:
        run_cli_mode(args)

if __name__ == "__main__":
    main() 
