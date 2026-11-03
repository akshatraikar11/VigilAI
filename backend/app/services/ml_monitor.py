"""
Vigil AI - Production ML Monitoring & Drift Detection
======================================================
Real-time monitoring for model performance, drift detection, and latency tracking.
"""

import time
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MLMonitor:
    """Production ML monitoring with drift detection and alerting."""
    
    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.1):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Performance tracking
        self.latencies = deque(maxlen=window_size)
        self.predictions = deque(maxlen=window_size)
        self.confidences = deque(maxlen=window_size)
        self.class_distribution = deque(maxlen=window_size)
        
        # Drift detection baselines (set after first 100 predictions)
        self.baseline_accuracy = None
        self.baseline_confidence = None
        self.baseline_class_dist = None
        self.baseline_set = False
        
        # Alert tracking
        self.alerts = []
        self.last_drift_check = datetime.now()
        
    def log_inference(self, text: str, prediction: str, confidence: float, 
                     latency_ms: float, actual_label: Optional[str] = None):
        """Log a single inference for monitoring."""
        
        # Store metrics
        self.latencies.append(latency_ms)
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.class_distribution.append(prediction)
        
        # Check if prediction was correct (if actual label provided)
        is_correct = actual_label == prediction if actual_label else None
        
        # Log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency_ms,
            'is_correct': is_correct
        }
        
        # Set baseline after enough data
        if len(self.predictions) >= 100 and not self.baseline_set:
            self._set_baseline()
            
        # Check for drift every 100 predictions
        if len(self.predictions) % 100 == 0:
            self._check_drift()
            
        return log_entry
    
    def _set_baseline(self):
        """Set performance baselines from initial data."""
        self.baseline_accuracy = 0.85  # Target accuracy
        self.baseline_confidence = np.mean(list(self.confidences))
        
        # Class distribution baseline
        class_counts = defaultdict(int)
        for pred in self.class_distribution:
            class_counts[pred] += 1
        total = sum(class_counts.values())
        self.baseline_class_dist = {k: v/total for k, v in class_counts.items()}
        
        self.baseline_set = True
        logger.info(f"Baseline set: accuracy={self.baseline_accuracy:.3f}, "
                   f"confidence={self.baseline_confidence:.3f}")
    
    def _check_drift(self):
        """Check for performance and data drift."""
        if not self.baseline_set:
            return
            
        current_time = datetime.now()
        drift_detected = False
        
        # Performance drift (confidence drop)
        current_confidence = np.mean(list(self.confidences)[-100:])  # Last 100
        confidence_drift = abs(current_confidence - self.baseline_confidence)
        
        if confidence_drift > self.alert_threshold:
            drift_detected = True
            alert = {
                'type': 'confidence_drift',
                'severity': 'high' if confidence_drift > 0.2 else 'medium',
                'message': f'Confidence drift detected: {current_confidence:.3f} vs baseline {self.baseline_confidence:.3f}',
                'timestamp': current_time.isoformat()
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
        
        # Latency drift
        current_p95_latency = np.percentile(list(self.latencies)[-100:], 95)
        if current_p95_latency > 300:  # 300ms threshold
            drift_detected = True
            alert = {
                'type': 'latency_drift',
                'severity': 'high',
                'message': f'Latency P95 exceeded: {current_p95_latency:.1f}ms',
                'timestamp': current_time.isoformat()
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
        
        # Class distribution drift
        current_class_counts = defaultdict(int)
        for pred in list(self.class_distribution)[-100:]:
            current_class_counts[pred] += 1
        current_total = sum(current_class_counts.values())
        current_dist = {k: v/current_total for k, v in current_class_counts.items()}
        
        # Calculate distribution drift (simplified KL divergence)
        dist_drift = 0
        for class_name in set(self.baseline_class_dist.keys()) | set(current_dist.keys()):
            p = self.baseline_class_dist.get(class_name, 0.001)
            q = current_dist.get(class_name, 0.001)
            dist_drift += p * np.log(p / q)
        
        if dist_drift > 0.1:  # Threshold for distribution drift
            drift_detected = True
            alert = {
                'type': 'distribution_drift',
                'severity': 'medium',
                'message': f'Class distribution drift detected: KL={dist_drift:.3f}',
                'timestamp': current_time.isoformat()
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
        
        self.last_drift_check = current_time
        return drift_detected
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        if not self.latencies:
            return {}
            
        return {
            'total_predictions': len(self.predictions),
            'avg_latency_ms': np.mean(self.latencies),
            'p95_latency_ms': np.percentile(self.latencies, 95),
            'p99_latency_ms': np.percentile(self.latencies, 99),
            'avg_confidence': np.mean(self.confidences),
            'recent_predictions': list(self.predictions)[-10:],
            'active_alerts': len([a for a in self.alerts if 
                                datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=24)]),
            'baseline_set': self.baseline_set
        }
    
    def get_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts if datetime.fromisoformat(a['timestamp']) > cutoff]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        metrics = self.get_metrics()
        metrics['export_timestamp'] = datetime.now().isoformat()
        metrics['recent_alerts'] = self.get_alerts(24)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")

# Global monitor instance
monitor = MLMonitor()

def log_inference(text: str, prediction: str, confidence: float, 
                 latency_ms: float, actual_label: Optional[str] = None):
    """Convenience function to log inference."""
    return monitor.log_inference(text, prediction, confidence, latency_ms, actual_label)

def get_metrics() -> Dict:
    """Get current metrics."""
    return monitor.get_metrics()

def check_production_health() -> Dict:
    """Check if system is healthy for production."""
    metrics = get_metrics()
    health_status = "healthy"
    issues = []
    
    # Check latency
    if metrics.get('p95_latency_ms', 0) > 300:
        health_status = "degraded"
        issues.append("P95 latency > 300ms")
    
    # Check confidence
    if metrics.get('avg_confidence', 0) < 0.7:
        health_status = "degraded"
        issues.append("Average confidence < 0.7")
    
    # Check alerts
    if metrics.get('active_alerts', 0) > 5:
        health_status = "critical"
        issues.append("Too many active alerts")
    
    return {
        'status': health_status,
        'issues': issues,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
