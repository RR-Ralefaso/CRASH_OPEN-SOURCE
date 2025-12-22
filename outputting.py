"""
Output management module for crash detection system.
Handles logging, file output, and data persistence.
"""
import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional


class DetectionLogger:
    """Logger for detection events and system activities."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = log_dir
        self.output_file = os.path.join(log_dir, "detections.csv")
        self.events_file = os.path.join(log_dir, "events.log")
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.output_file):
            self._init_csv_file()
        
        # Session tracking
        self.session_start = None
        self.detection_count = 0
        
    def _init_csv_file(self):
        """Initialize CSV file with headers."""
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'session_id',
                'objects_detected',
                'cars',
                'trucks',
                'persons',
                'potential_crash',
                'avg_confidence',
                'frame_number'
            ])
    
    def log_start_session(self):
        """Log the start of a new detection session."""
        self.session_start = datetime.now()
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        event_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " \
                   f"Session STARTED - ID: {session_id}"
        
        self._write_event(event_msg)
        print(event_msg)
        
        # Reset counters
        self.detection_count = 0
        
        return session_id
    
    def log_detection(self, detection_info: Dict[str, Any], frame_number: Optional[int] = None):
        """
        Log a detection event.
        
        Args:
            detection_info: Dictionary containing detection details
            frame_number: Current frame number (optional)
        """
        if not detection_info:
            return
        
        # Generate session ID if not started
        if self.session_start is None:
            session_id = self.log_start_session()
        else:
            session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Prepare CSV row
        row = [
            detection_info.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            session_id,
            detection_info.get('objects_detected', 0),
            detection_info.get('cars', 0),
            detection_info.get('trucks', 0),
            detection_info.get('persons', 0),
            'YES' if detection_info.get('potential_crash', False) else 'NO',
            f"{detection_info.get('confidence_avg', 0.0):.3f}",
            frame_number if frame_number is not None else ''
        ]
        
        # Write to CSV
        with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Log significant events
        if detection_info.get('potential_crash', False):
            event_msg = f"[{detection_info['timestamp']}] " \
                       f"POTENTIAL CRASH DETECTED! Objects: {detection_info['objects_detected']}, " \
                       f"Cars: {detection_info.get('cars', 0)}, " \
                       f"Trucks: {detection_info.get('trucks', 0)}"
            self._write_event(event_msg)
            print(f"⚠️  {event_msg}")
        
        self.detection_count += 1
    
    def log_event(self, message: str, level: str = "INFO"):
        """
        Log a general event or system message.
        
        Args:
            message: Event message
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_msg = f"[{timestamp}] [{level}] {message}"
        
        self._write_event(event_msg)
        
        if level == "ERROR":
            print(f"❌ {event_msg}")
        elif level == "WARNING":
            print(f"⚠️  {event_msg}")
        else:
            print(f"ℹ️  {event_msg}")
    
    def log_end_session(self):
        """Log the end of a detection session."""
        if self.session_start is None:
            return
        
        duration = datetime.now() - self.session_start
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        event_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " \
                   f"Session ENDED - Duration: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}, " \
                   f"Detections: {self.detection_count}"
        
        self._write_event(event_msg)
        print(event_msg)
        
        # Generate session summary
        self._generate_session_summary()
    
    def _write_event(self, message: str):
        """Write an event message to the log file."""
        with open(self.events_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def _generate_session_summary(self):
        """Generate a summary JSON file for the session."""
        if self.session_start is None:
            return
        
        summary_file = os.path.join(self.log_dir, f"session_{self.session_start.strftime('%Y%m%d_%H%M%S')}_summary.json")
        
        summary = {
            'session_id': self.session_start.strftime("%Y%m%d_%H%M%S"),
            'start_time': self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'duration_seconds': (datetime.now() - self.session_start).total_seconds(),
            'total_detections': self.detection_count,
            'log_files': {
                'detections': os.path.basename(self.output_file),
                'events': os.path.basename(self.events_file)
            }
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about logged detections."""
        stats = {
            'total_sessions': 0,
            'total_detections': 0,
            'potential_crashes': 0,
            'last_session': None
        }
        
        if not os.path.exists(self.output_file):
            return stats
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    stats['total_detections'] = len(rows)
                    stats['potential_crashes'] = sum(1 for row in rows if row.get('potential_crash', '').upper() == 'YES')
                    
                    # Count unique sessions
                    sessions = set(row.get('session_id', '') for row in rows)
                    stats['total_sessions'] = len(sessions)
                    
                    # Get most recent session
                    if sessions:
                        stats['last_session'] = max(sessions)
        
        except Exception as e:
            print(f"Error reading stats: {e}")
        
        return stats
    
    def export_detections_json(self, output_path: Optional[str] = None):
        """
        Export detections to JSON format.
        
        Args:
            output_path: Custom output path (default: logs/detections_export.json)
        """
        if not os.path.exists(self.output_file):
            return
        
        if output_path is None:
            output_path = os.path.join(self.log_dir, "detections_export.json")
        
        try:
            with open(self.output_file, 'r', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                detections = list(reader)
            
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(detections, json_file, indent=2)
            
            print(f"Detections exported to: {output_path}")
            return output_path
        
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return None


# Convenience functions for backward compatibility
def reset_output_file():
    """Reset the output file (creates new log session)."""
    logger = DetectionLogger()
    return logger.log_start_session()

def write_detection_output(detection_info: str):
    """Write detection output (legacy function)."""
    logger = DetectionLogger()
    logger.log_event(detection_info, "DETECTION")