import sys
import os
import cv2
import numpy as np
import datetime
import time
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QMessageBox, QFrame,
    QGridLayout, QGroupBox, QScrollArea, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QSlider, QCheckBox, QLineEdit, QSpacerItem
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fall_detector import FallDetector

class ModernStyle:
    """Modern UI style definitions for the application."""
    
    # Color scheme
    DARK_PRIMARY = "#1a73e8"       # Google Blue
    LIGHT_PRIMARY = "#4285f4"
    ACCENT = "#fbbc05"             # Yellow
    DANGER = "#ea4335"             # Red
    SUCCESS = "#34a853"            # Green
    BG_DARK = "#202124"
    BG_LIGHT = "#f8f9fa"
    TEXT_DARK = "#3c4043"
    TEXT_LIGHT = "#e8eaed"
    CARD_BG = "#ffffff"
    
    # Stylesheet templates
    MAIN_STYLESHEET = """
        QMainWindow, QDialog {
            background-color: %s;
        }
        QTabWidget {
            background-color: %s;
        }
        QTabWidget::pane {
            border: none;
            background-color: %s;
        }
        QTabBar::tab {
            background-color: %s;
            color: %s;
            padding: 8px 15px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
        QTabBar::tab:selected {
            background-color: %s;
            color: %s;
            border-bottom: 3px solid %s;
        }
        QWidget {
            font-family: 'Segoe UI', Arial;
            font-size: 10pt;
        }
        QLabel {
            color: %s;
        }
        QPushButton {
            background-color: %s;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: %s;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        QGroupBox {
            background-color: %s;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
            margin-top: 15px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 5px;
            color: %s;
        }
    """ % (
        BG_LIGHT, BG_LIGHT, BG_LIGHT,  # Main colors
        BG_LIGHT, TEXT_DARK,           # Tab normal
        CARD_BG, DARK_PRIMARY, DARK_PRIMARY,  # Tab selected
        TEXT_DARK,                     # Label text
        LIGHT_PRIMARY, DARK_PRIMARY,   # Button colors
        CARD_BG,                       # GroupBox background
        DARK_PRIMARY                   # GroupBox title
    )
    
    # Button style variations
    PRIMARY_BTN = """
        QPushButton {
            background-color: %s;
            color: white;
        }
        QPushButton:hover {
            background-color: %s;
        }
    """ % (DARK_PRIMARY, LIGHT_PRIMARY)
    
    DANGER_BTN = """
        QPushButton {
            background-color: %s;
            color: white;
        }
        QPushButton:hover {
            background-color: #ff6b6b;
        }
    """ % (DANGER)
    
    SUCCESS_BTN = """
        QPushButton {
            background-color: %s;
            color: white;
        }
        QPushButton:hover {
            background-color: #4ad66d;
        }
    """ % (SUCCESS)

class FallDetectionDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set window title and size
        self.setWindowTitle("Fall Detection System Dashboard")
        self.resize(1280, 800)
        
        # Apply modern styles
        self.setStyleSheet(ModernStyle.MAIN_STYLESHEET)
        
        # Dashboard state variables
        self.is_running = False
        self.camera_source = 0  # Default camera source
        self.video_source = None
        self.video_paused = False
        self.detection_enabled = True
        self.fall_detector = None
        self.cap = None
        
        # Statistics
        self.stats = {
            "total_people_detected": 0,
            "total_falls_detected": 0,
            "alarm_counts": {
                "transformer_fall": 0,
            },
            "detection_history": []  # List of (timestamp, fall_type) tuples
        }
        
        # Load configuration
        self.config = {
            "model_path": "yolov12n1.pt",
            "confidence": 0.5,
            "sound_alerts": True,
            "auto_save_falls": True,
            "output_dir": "fall_snapshots",
            "fall_cooldown": 5.0  # Cooldown in seconds for the same person
        }
        
        # Initialize tracking set for fallen IDs
        self.tracked_fallen_ids = set()
        
        # Track when each person fell (for cooldown)
        self.fallen_timestamps = {}
        
        # Last snapshot time to avoid saving too many images
        self.last_snapshot_time = 0
        self.snapshot_cooldown = 2.0  # seconds
        
        # Create the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Setup the UI components
        self.setup_ui()
        QApplication.instance().aboutToQuit.connect(self.stop_detection)
        
    def setup_ui(self):
        """Set up the main user interface."""
        # Create a tab widget
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # Create tabs
        self.dashboard_tab = QWidget()
        self.settings_tab = QWidget()
        self.history_tab = QWidget()
        
        # Add tabs to the tab widget
        self.tab_widget.addTab(self.dashboard_tab, "Dashboard")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.history_tab, "History")
        
        # Setup each tab
        self.setup_dashboard_tab()
        self.setup_settings_tab()
        self.setup_history_tab()
        
        # Add status bar at the bottom
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready to start detection")

    def setup_dashboard_tab(self):
        """Set up the main dashboard tab."""
        # Main layout for dashboard tab
        layout = QHBoxLayout(self.dashboard_tab)
        
        # Create left panel (video feed and controls)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create right panel (statistics and charts)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add panels to main layout
        layout.addWidget(left_panel, 3)  # 3:2 ratio
        layout.addWidget(right_panel, 2)
        
        # Video display frame
        video_group = QGroupBox("Video Feed")
        video_layout = QVBoxLayout(video_group)
        video_layout.setContentsMargins(10, 20, 10, 10)
        
        # Video canvas
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        video_layout.addWidget(self.video_label)
        
        # Add video frame to left panel
        left_layout.addWidget(video_group)
        
        # Controls group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 20, 10, 10)
        
        # Source selection
        source_layout = QHBoxLayout()
        source_layout.addWidget(QLabel("Video Source:"))
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems(["Webcam (0)", "Webcam (1)", "Webcam (2)"])
        self.camera_combo.setCurrentIndex(0)
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        source_layout.addWidget(self.camera_combo)
        
        self.browse_btn = QPushButton("Browse Video File")
        self.browse_btn.clicked.connect(self.browse_video_file)
        source_layout.addWidget(self.browse_btn)
        source_layout.addStretch(1)
        
        controls_layout.addLayout(source_layout)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Start/Stop button
        self.start_stop_btn = QPushButton("Start Detection")
        self.start_stop_btn.clicked.connect(self.toggle_detection)
        self.start_stop_btn.setStyleSheet(ModernStyle.PRIMARY_BTN)
        self.start_stop_btn.setMinimumWidth(150)
        buttons_layout.addWidget(self.start_stop_btn)
        
        # Pause/Resume button
        self.pause_resume_btn = QPushButton("Pause")
        self.pause_resume_btn.clicked.connect(self.toggle_pause)
        self.pause_resume_btn.setEnabled(False)
        self.pause_resume_btn.setMinimumWidth(100)
        buttons_layout.addWidget(self.pause_resume_btn)
        
        # Reset stats button
        self.reset_stats_btn = QPushButton("Reset Stats")
        self.reset_stats_btn.clicked.connect(self.reset_statistics)
        self.reset_stats_btn.setStyleSheet(ModernStyle.DANGER_BTN)
        self.reset_stats_btn.setMinimumWidth(100)
        buttons_layout.addWidget(self.reset_stats_btn)
        
        # Settings button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        self.settings_btn.setMinimumWidth(100)
        buttons_layout.addWidget(self.settings_btn)
        
        buttons_layout.addStretch(1)
        controls_layout.addLayout(buttons_layout)
        
        # Add controls to left panel
        left_layout.addWidget(controls_group)
        
        # Statistics group
        stats_group = QGroupBox("Detection Statistics")
        stats_layout = QGridLayout(stats_group)
        stats_layout.setContentsMargins(10, 20, 10, 10)
        
        # People detected
        stats_layout.addWidget(QLabel("People Detected:"), 0, 0)
        self.people_count_label = QLabel("0")
        self.people_count_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.people_count_label, 0, 1)
        
        # Falls detected
        stats_layout.addWidget(QLabel("Falls Detected:"), 1, 0)
        self.fall_count_label = QLabel("0")
        self.fall_count_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.fall_count_label, 1, 1)
        
        # FPS
        stats_layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_label = QLabel("0.0")
        self.fps_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(self.fps_label, 2, 1)
        
        # Classification heading
        alarm_counts_label = QLabel("Fall Classification:")
        alarm_counts_label.setStyleSheet("font-weight: bold;")
        stats_layout.addWidget(alarm_counts_label, 3, 0, 1, 2)
        
        # Binary fall classification
        self.fall_type_labels = {}
        alarm_counts = [
            ("transformer_fall", "Transformer Fall"),
        ]
        
        for i, (fall_type, display_name) in enumerate(alarm_counts):
            stats_layout.addWidget(QLabel(f"   {display_name}:"), i+4, 0)
            self.fall_type_labels[fall_type] = QLabel("0")
            stats_layout.addWidget(self.fall_type_labels[fall_type], i+4, 1)
        
        # Add statistics to right panel
        right_layout.addWidget(stats_group)
        
        # Charts
        # Pie chart
        pie_group = QGroupBox("Fall Events")
        pie_layout = QVBoxLayout(pie_group)
        pie_layout.setContentsMargins(10, 20, 10, 10)
        
        self.pie_figure = Figure(figsize=(4, 4), dpi=100)
        self.pie_canvas = FigureCanvas(self.pie_figure)
        self.pie_ax = self.pie_figure.add_subplot(111)
        pie_layout.addWidget(self.pie_canvas)
        
        # Timeline chart
        timeline_group = QGroupBox("Detection Timeline")
        timeline_layout = QVBoxLayout(timeline_group)
        timeline_layout.setContentsMargins(10, 20, 10, 10)
        
        self.timeline_figure = Figure(figsize=(4, 3), dpi=100)
        self.timeline_canvas = FigureCanvas(self.timeline_figure)
        self.timeline_ax = self.timeline_figure.add_subplot(111)
        timeline_layout.addWidget(self.timeline_canvas)
        
        # Add charts to right panel
        right_layout.addWidget(pie_group)
        right_layout.addWidget(timeline_group)
        
        # Initialize charts
        self.update_pie_chart()
        self.update_timeline_chart()

    def setup_settings_tab(self):
        """Set up the settings tab."""
        # Main layout for settings tab
        layout = QVBoxLayout(self.settings_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QGridLayout(model_group)
        model_layout.setContentsMargins(15, 25, 15, 15)
        model_layout.setSpacing(10)
        
        # Model path
        model_layout.addWidget(QLabel("YOLOv12 Model:"), 0, 0)
        
        model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit(self.config["model_path"])
        model_path_layout.addWidget(self.model_path_edit)
        
        browse_model_btn = QPushButton("Browse")
        browse_model_btn.clicked.connect(self.browse_model_file)
        model_path_layout.addWidget(browse_model_btn)
        
        model_layout.addLayout(model_path_layout, 0, 1)
        
        # Confidence threshold
        model_layout.addWidget(QLabel("Confidence Threshold:"), 1, 0)
        
        conf_layout = QHBoxLayout()
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(1)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(int(self.config["confidence"] * 100))
        self.confidence_slider.setTickPosition(QSlider.TicksBelow)
        self.confidence_slider.setTickInterval(10)
        conf_layout.addWidget(self.confidence_slider)
        
        self.confidence_label = QLabel(f"{self.config['confidence']:.2f}")
        conf_layout.addWidget(self.confidence_label)
        
        self.confidence_slider.valueChanged.connect(self.update_confidence_label)
        model_layout.addLayout(conf_layout, 1, 1)
        
        # Add model settings to main layout
        layout.addWidget(model_group)
        
        detection_group = QGroupBox("Fall Classification")
        detection_layout = QGridLayout(detection_group)
        detection_layout.addWidget(QLabel("Fall probability threshold: 0.9 (fixed)"), 0, 0)
        layout.addWidget(detection_group)

        # Notification settings
        notification_group = QGroupBox("Notification Settings")
        notification_layout = QGridLayout(notification_group)
        notification_layout.setContentsMargins(15, 25, 15, 15)
        notification_layout.setSpacing(10)
        
        # Sound alerts
        self.sound_alerts_check = QCheckBox("Enable Sound Alerts")
        self.sound_alerts_check.setChecked(self.config.get("sound_alerts", True))
        notification_layout.addWidget(self.sound_alerts_check, 0, 0, 1, 2)
        
        # Auto-save falls
        self.auto_save_falls_check = QCheckBox("Automatically Save Fall Snapshots")
        self.auto_save_falls_check.setChecked(self.config.get("auto_save_falls", True))
        notification_layout.addWidget(self.auto_save_falls_check, 1, 0, 1, 2)
        
        # Output directory
        notification_layout.addWidget(QLabel("Output Directory:"), 2, 0)
        
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.config.get("output_dir", "fall_snapshots"))
        output_dir_layout.addWidget(self.output_dir_edit)
        
        browse_dir_btn = QPushButton("Browse")
        browse_dir_btn.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(browse_dir_btn)
        
        notification_layout.addLayout(output_dir_layout, 2, 1)
        
        # Add notification settings to main layout
        layout.addWidget(notification_group)
        
        # Save settings button
        save_layout = QHBoxLayout()
        save_layout.addStretch(1)
        
        save_settings_btn = QPushButton("Save Settings")
        save_settings_btn.setStyleSheet(ModernStyle.SUCCESS_BTN)
        save_settings_btn.setMinimumWidth(150)
        save_settings_btn.clicked.connect(self.save_settings)
        save_layout.addWidget(save_settings_btn)
        
        layout.addLayout(save_layout)
        layout.addStretch(1)

    def setup_history_tab(self):
        """Set up the history tab."""
        # Main layout for history tab
        layout = QVBoxLayout(self.history_tab)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Toolbar
        toolbar_layout = QHBoxLayout()
        
        # Filter options
        toolbar_layout.addWidget(QLabel("Filter by:"))
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "Transformer Fall"])
        self.filter_combo.currentIndexChanged.connect(self.filter_history)
        toolbar_layout.addWidget(self.filter_combo)
        
        toolbar_layout.addStretch(1)
        
        # Export button
        export_btn = QPushButton("Export History")
        export_btn.clicked.connect(self.export_history)
        toolbar_layout.addWidget(export_btn)
        
        layout.addLayout(toolbar_layout)
        
        # History tree widget
        self.history_tree = QTreeWidget()
        self.history_tree.setColumnCount(3)
        self.history_tree.setHeaderLabels(["Time", "Fall Type", "Details"])
        self.history_tree.setAlternatingRowColors(True)
        self.history_tree.setStyleSheet("""
            QTreeWidget {
                background-color: white;
                alternate-background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                font-size: 10pt;
            }
            QHeaderView::section {
                background-color: #4285f4;
                color: white;
                font-weight: bold;
                padding: 6px;
                border: none;
            }
        """)
        
        # Set column widths
        header = self.history_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        
        layout.addWidget(self.history_tree)

    # UI utility methods
    def update_confidence_label(self, value):
        """Update confidence threshold label when slider changes."""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
    
    
    
    # File and directory browsing methods
    def browse_model_file(self):
        """Open file dialog to select a model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLOv12 Model File",
            "",
            "PyTorch models (*.pt);;All Files (*)"
        )
        
        if file_path:
            self.model_path_edit.setText(file_path)
    
    def browse_output_dir(self):
        """Open directory dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Fall Snapshots"
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def browse_video_file(self):
        """Open file dialog to select a video file."""
        if self.is_running:
            QMessageBox.information(self, "Info", "Please stop detection before changing the video source.")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_path:
            self.video_source = file_path
            # Update camera combo box to show video file name
            self.camera_combo.setCurrentText(os.path.basename(file_path))
    
    # Settings management
    def save_settings(self):
        """Save the settings to the configuration."""
        # Update configuration with current settings
        self.config["model_path"] = self.model_path_edit.text()
        self.config["confidence"] = self.confidence_slider.value() / 100.0
        self.config["sound_alerts"] = self.sound_alerts_check.isChecked()
        self.config["auto_save_falls"] = self.auto_save_falls_check.isChecked()
        self.config["output_dir"] = self.output_dir_edit.text()
        
        # If detection is running, apply the changes to the fall detector
        if self.fall_detector:
            self.fall_detector.confidence = self.config["confidence"]
        
        # Save to a configuration file (optional)
        try:
            os.makedirs(os.path.dirname(os.path.abspath("config/settings.json")), exist_ok=True)
            with open("config/settings.json", "w") as f:
                json.dump(self.config, f, indent=4)
            QMessageBox.information(self, "Settings Saved", "Configuration has been saved successfully.")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save configuration to file: {str(e)}")
        
        # Switch back to main tab
        self.tab_widget.setCurrentIndex(0)
    
    # Camera source management
    def on_camera_changed(self, index):
        """Handle camera source selection changes."""
        if self.is_running:
            QMessageBox.information(self, "Info", "Please stop detection before changing the camera source.")
            return
        
        selection = self.camera_combo.currentText()
        if selection.startswith("Webcam"):
            # Extract camera number from string like "Webcam (0)"
            try:
                camera_id = int(selection.split("(")[1].split(")")[0])
                self.camera_source = camera_id
                self.video_source = None
            except:
                QMessageBox.critical(self, "Error", "Invalid camera selection.")
    
    # Video processing methods
    def toggle_detection(self):
        """Start or stop the fall detection process."""
        if self.is_running:
            self.stop_detection()
        else:
            # Start detection
            self.start_detection()

    def stop_detection(self):
        """Stop GUI-thread frame callbacks before disposing of their resources."""
        self.is_running = False
        if hasattr(self, 'video_timer'):
            self.video_timer.stop()
        # QTimer and process_frame run synchronously on the GUI thread.
        # FallDetector.close also waits for processing by any other caller.
        try:
            if self.fall_detector is not None:
                self.fall_detector.close()
                self.fall_detector = None
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        self.start_stop_btn.setText("Start Detection")
        self.pause_resume_btn.setEnabled(False)
        self.status_bar.showMessage("Ready to start detection")

    def closeEvent(self, event):
        self.stop_detection()
        super().closeEvent(event)
    
    def toggle_pause(self):
        """Pause or resume the video processing."""
        if self.video_paused:
            # Resume
            self.video_paused = False
            self.pause_resume_btn.setText("Pause")
            self.status_bar.showMessage("Running")
        else:
            # Pause
            self.video_paused = True
            self.pause_resume_btn.setText("Resume")
            self.status_bar.showMessage("Paused")
    
    def start_detection(self):
        """Initialize and start the fall detection process."""
        if self.is_running:
            return
        try:
            self.stop_detection()
            # Initialize the fall detector if not already created
            if not self.fall_detector:
                self.fall_detector = FallDetector(
                    model_path=self.config["model_path"],
                    confidence=self.config["confidence"]
                )
                
            
            # Open video source
            if self.video_source:
                # Video file
                self.cap = cv2.VideoCapture(self.video_source)
            else:
                # Camera - try without DSHOW first
                self.cap = cv2.VideoCapture(self.camera_source)
            
            if not self.cap.isOpened():
                # Try with DSHOW on Windows as fallback
                try:
                    if os.name == 'nt':  # Windows
                        self.cap.release()
                        self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_DSHOW)
                except:
                    pass
                    
                if not self.cap.isOpened():
                    self.stop_detection()
                    QMessageBox.critical(self, "Error", "Could not open video source.")
                    return
            
            # Update UI state
            self.is_running = True
            self.video_paused = False
            # ByteTrack IDs belong to this new detector session.
            self.tracked_fallen_ids.clear()
            self.fallen_timestamps.clear()
            self.start_stop_btn.setText("Stop Detection")
            self.pause_resume_btn.setEnabled(True)
            self.pause_resume_btn.setText("Pause")
            self.status_bar.showMessage("Running")
            
            # Initialize FPS counter
            self.frame_count = 0
            self.last_fps_update = time.time()
            self.fps = 0.0
            
            # Create and start the video processing timer
            if not hasattr(self, 'video_timer'):
                self.video_timer = QTimer(self)
                self.video_timer.timeout.connect(self.process_frame)
            self.video_timer.start(30)  # 30ms interval (~33 FPS)
            
        except Exception as e:
            self.stop_detection()
            QMessageBox.critical(self, "Error", f"Failed to start detection: {str(e)}")
            self.is_running = False
            import traceback
            traceback.print_exc()
    
    def process_frame(self):
        """Process a single video frame."""
        if not self.is_running or not self.cap:
            return
            
        if not self.video_paused:
            # Read a frame
            ret, frame = self.cap.read()
            if not ret:
                if self.video_source:
                    # Video file ended
                    self.stop_detection()
                    QMessageBox.information(self, "Info", "Video playback completed.")
                    return
                else:
                    # Camera frame read error - try again next time
                    return
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            time_diff = current_time - self.last_fps_update
            
            if time_diff >= 1.0:
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_fps_update = current_time
                # Update FPS display
                self.fps_label.setText(f"{self.fps:.1f}")
            
            # Process the frame for fall detection
            if self.detection_enabled:
                try:
                    output_frame, fall_detected, fall_data = self.fall_detector.process_frame(frame)
                except Exception as e:
                    self.stop_detection()
                    QMessageBox.critical(self, "Error", f"Detection failed: {str(e)}")
                    return
                
                # Add FPS display to the frame
                fps_text = f"FPS: {self.fps:.1f}"
                cv2.putText(
                    output_frame, 
                    fps_text, 
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                
                # Update statistics
                self.update_statistics(fall_data)
                
                # Display the processed frame
                self.display_frame(output_frame)
            else:
                # Display the original frame without detection
                self.display_frame(frame)
    
    def display_frame(self, frame):
        """Display a frame in the video label."""
        # Resize the frame to fit the label
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        
        # Convert to RGB format for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage and then to QPixmap
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Scale pixmap to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(
            self.video_label.width(), 
            self.video_label.height(),
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        # Update label
        self.video_label.setPixmap(pixmap)
        self.current_frame = frame  # Store for snapshot saving
    
    # Statistics and visualization methods
    def update_statistics(self, fall_data):
        """Update the statistics with new detection data."""
        # Update people count
        if "person_count" in fall_data:
            current_people = fall_data["person_count"]
            self.stats["total_people_detected"] = max(self.stats["total_people_detected"], current_people)
            self.people_count_label.setText(str(self.stats["total_people_detected"]))
        
        for person_id in fall_data.get("expired_ids", []):
            self.fallen_timestamps.pop(person_id, None)
            self.tracked_fallen_ids.discard(person_id)

        # Cached positive scores remain visible, but only fresh positive
        # inferences can create events. Cooldown never affects detector updates.
        event_ids = set(fall_data.get("fall_event_ids", []))
        if not event_ids:
            return
        current_time = time.time()
        new_fallen_ids = set()
        for person_id in event_ids:
            if (person_id not in self.fallen_timestamps or
                    current_time - self.fallen_timestamps[person_id]
                    > self.config.get("fall_cooldown", 5.0)):
                new_fallen_ids.add(person_id)
                self.fallen_timestamps[person_id] = current_time
        if new_fallen_ids:
            self._process_new_falls("transformer_fall", new_fallen_ids, current_time)

    def update_pie_chart(self):
        """Update the pie chart with current fall type data."""
        # Clear previous chart
        self.pie_ax.clear()
        
        # Get fall type data
        alarm_counts = list(self.stats["alarm_counts"].keys())
        fall_counts = [self.stats["alarm_counts"][fall_type] for fall_type in alarm_counts]
        
        # Format labels
        fall_type_labels = [t.replace("_", " ").title() for t in alarm_counts]
        
        # Check if there's any data to display
        if sum(fall_counts) == 0:
            self.pie_ax.text(
                0.5, 0.5, 
                "No falls detected", 
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.pie_ax.transAxes,
                fontsize=12
            )
        else:
            # Create the pie chart
            wedges, texts, autotexts = self.pie_ax.pie(
                fall_counts, 
                labels=fall_type_labels, 
                autopct='%1.1f%%',
                startangle=90,
                colors=['#4285f4', '#34a853', '#fbbc05', '#ea4335']  # Google colors
            )
            
            # Style the chart
            for text in texts:
                text.set_fontsize(9)
            for autotext in autotexts:
                autotext.set_fontsize(9)
                autotext.set_fontweight('bold')
        
        # Adjust layout and update
        self.pie_figure.tight_layout()
        self.pie_canvas.draw()
    
    def update_timeline_chart(self):
        """Update the timeline chart with fall detections over time."""
        # Clear previous chart
        self.timeline_ax.clear()
        
        # Check if we have any data
        if not self.stats["detection_history"]:
            self.timeline_ax.text(
                0.5, 0.5, 
                "No detection history", 
                horizontalalignment='center',
                verticalalignment='center',
                transform=self.timeline_ax.transAxes,
                fontsize=12
            )
        else:
            # Extract timestamps and convert to datetime objects
            timestamps = [entry[0] for entry in self.stats["detection_history"]]
            times = [datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in timestamps]
            
            # Count falls by time (binned by minute)
            time_counts = {}
            for t in times:
                # Round to nearest minute to bin
                minute_time = t.replace(second=0)
                if minute_time in time_counts:
                    time_counts[minute_time] += 1
                else:
                    time_counts[minute_time] = 1
            
            # Sort times for plotting
            sorted_times = sorted(time_counts.keys())
            counts = [time_counts[t] for t in sorted_times]
            
            # Plot time series
            self.timeline_ax.plot(sorted_times, counts, marker='o', linestyle='-', 
                             color=ModernStyle.DARK_PRIMARY)
            self.timeline_ax.set_title('Falls Over Time')
            self.timeline_ax.set_ylabel('Count')
            
            # Format x-axis for better readability
            self.timeline_ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
            self.timeline_ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout and update
        self.timeline_figure.tight_layout()
        self.timeline_canvas.draw()
    
    # History management methods
    def filter_history(self):
        """Filter history based on selected fall type."""
        filter_value = self.filter_combo.currentText()
        
        # Clear the current display
        self.history_tree.clear()
        
        # Add all entries that match the filter
        for timestamp, fall_type, details in self.get_filtered_history(filter_value):
            item = QTreeWidgetItem([timestamp, fall_type, details])
            self.history_tree.addTopLevelItem(item)

    def get_filtered_history(self, filter_value):
        """Get history entries matching the filter."""
        filtered_entries = []
        
        for item in self.stats["detection_history"]:
            timestamp, fall_type = item
            
            # Format the fall type for display
            display_fall_type = "Unknown"
            if fall_type:
                display_fall_type = fall_type.replace("_", " ").title()
            
            # Details
            details = "Fall detected with critical points identified"
            
            # Apply filter
            if filter_value == "All" or display_fall_type == filter_value:
                filtered_entries.append((timestamp, display_fall_type, details))
        
        return filtered_entries

    def export_history(self):
        """Export detection history to a CSV file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Detection History",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write("Timestamp,Fall Type,Details\n")
                    
                    for timestamp, fall_type, details in self.get_filtered_history(self.filter_combo.currentText()):
                        f.write(f'"{timestamp}","{fall_type}","{details}"\n')
                
                QMessageBox.information(self, "Export Complete", f"History data has been exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export history: {str(e)}")
    
    # Utility methods
    def play_alert_sound(self):
        """Play a sound alert when a fall is detected."""
        try:
            # Only import if needed
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        except:
            # If winsound not available (non-Windows), try another approach
            try:
                import sys
                from subprocess import call
                
                if sys.platform == 'darwin':  # macOS
                    call(['afplay', '/System/Library/Sounds/Sosumi.aiff'])
                elif sys.platform.startswith('linux'):  # Linux
                    call(['paplay', '/usr/share/sounds/freedesktop/stereo/dialog-warning.oga'])
            except:
                # If all else fails, print to console
                print('\a')  # ASCII bell
    
    def save_fall_snapshot(self):
        """Save the current frame as an image when a fall is detected."""
        try:
            # Make sure output directory exists
            output_dir = self.config.get("output_dir", "fall_snapshots")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fall_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save the image if we have a current frame
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                cv2.imwrite(filepath, self.current_frame)
                print(f"Fall snapshot saved to: {filepath}")
        except Exception as e:
            print(f"Error saving fall snapshot: {e}")
    
    def reset_statistics(self):
        """Reset all fall detection statistics."""
        reply = QMessageBox.question(
            self, 
            "Reset Statistics", 
            "Are you sure you want to reset all statistics?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset statistics
            self.stats = {
                "total_people_detected": 0,
                "total_falls_detected": 0,
                "alarm_counts": {
                    "transformer_fall": 0,
                },
                "detection_history": []
            }
            
            # Reset tracked fallen IDs
            self.tracked_fallen_ids = set()
            
            # Update UI
            self.people_count_label.setText("0")
            self.fall_count_label.setText("0")
            
            for fall_type in self.fall_type_labels:
                self.fall_type_labels[fall_type].setText("0")
            
            # Clear detection tree
            self.history_tree.clear()
            
            # Update charts
            self.update_pie_chart()
            self.update_timeline_chart()
            
            QMessageBox.information(self, "Reset Complete", "Statistics have been reset.")

    def _process_new_falls(self, fall_type, new_fallen_ids, current_time):
        """Process newly detected falls."""
        # Increment counter by number of new falls
        self.stats["total_falls_detected"] += len(new_fallen_ids)
        self.fall_count_label.setText(str(self.stats["total_falls_detected"]))
        
        # Play sound alert if enabled
        if self.config.get("sound_alerts", True):
            self.play_alert_sound()
        
        # Save fall snapshot if enabled and not in cooldown
        if (self.config.get("auto_save_falls", True) and 
            current_time - self.last_snapshot_time > self.snapshot_cooldown):
            self.save_fall_snapshot()
            self.last_snapshot_time = current_time
        
        # Update binary event count
        self.stats["alarm_counts"][fall_type] += len(new_fallen_ids)
        self.fall_type_labels[fall_type].setText(str(self.stats["alarm_counts"][fall_type]))
        
        # Update tracking info
        self.tracked_fallen_ids.update(new_fallen_ids)
        
        # Add to detection history
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format fall type for display
        display_fall_type = fall_type.replace("_", " ").title()
        
        # Create details string
        details = f"Fall detected: {display_fall_type}"
        
        # Add to history tree (most recent at top)
        item = QTreeWidgetItem([timestamp, display_fall_type, details])
        self.history_tree.insertTopLevelItem(0, item)
        
        # Add to history for charts
        self.stats["detection_history"].append((timestamp, fall_type))
        
        # Update the charts
        self.update_pie_chart()
        self.update_timeline_chart()
        
        # Print confirmation
        print(f"Recorded new fall of type: {fall_type}")


def run_dashboard_mode(args):
    """Run the fall detection system in dashboard mode."""
    # Switch from tkinter to PyQt5
    app = QApplication(sys.argv)
    window = FallDetectionDashboard()
    try:
        window.show()
        sys.exit(app.exec_())
    finally:
        window.stop_detection()

if __name__ == "__main__":
    run_dashboard_mode(None) 
