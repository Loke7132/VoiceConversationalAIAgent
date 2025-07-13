import time
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Utility class for tracking performance metrics."""
    
    def __init__(self):
        self.start_times: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}
    
    def start(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation to time
        """
        self.start_times[operation] = time.time()
        logger.debug(f"Started timing operation: {operation}")
    
    def end(self, operation: str) -> float:
        """
        End timing an operation and calculate duration.
        
        Args:
            operation: Name of the operation to end timing for
            
        Returns:
            Duration in seconds
        """
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        end_time = time.time()
        duration = end_time - self.start_times[operation]
        self.durations[operation] = duration
        
        logger.debug(f"Completed timing operation: {operation} - {duration:.4f}s")
        return duration
    
    def get_duration(self, operation: str) -> float:
        """
        Get the duration of a completed operation.
        
        Args:
            operation: Name of the operation
            
        Returns:
            Duration in seconds, or 0.0 if operation not found
        """
        return self.durations.get(operation, 0.0)
    
    def get_all_durations(self) -> Dict[str, float]:
        """
        Get all recorded durations.
        
        Returns:
            Dictionary of operation names to durations
        """
        return self.durations.copy()
    
    def reset(self) -> None:
        """Reset all timing data."""
        self.start_times.clear()
        self.durations.clear()
        logger.debug("Reset performance tracker")
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get a summary of all performance metrics.
        
        Returns:
            Dictionary with total time, operation count, and average time
        """
        if not self.durations:
            return {
                "total_time": 0.0,
                "operation_count": 0,
                "average_time": 0.0
            }
        
        total_time = sum(self.durations.values())
        operation_count = len(self.durations)
        average_time = total_time / operation_count if operation_count > 0 else 0.0
        
        return {
            "total_time": total_time,
            "operation_count": operation_count,
            "average_time": average_time
        }
    
    def log_summary(self) -> None:
        """Log a summary of all performance metrics."""
        summary = self.get_summary()
        
        logger.info(f"Performance Summary:")
        logger.info(f"  Total Time: {summary['total_time']:.4f}s")
        logger.info(f"  Operations: {summary['operation_count']}")
        logger.info(f"  Average Time: {summary['average_time']:.4f}s")
        
        if self.durations:
            logger.info("  Individual Operations:")
            for operation, duration in self.durations.items():
                logger.info(f"    {operation}: {duration:.4f}s")


class GlobalPerformanceTracker:
    """Global performance tracker for application-wide metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.total_request_time = 0.0
        self.operation_counts: Dict[str, int] = {}
        self.operation_times: Dict[str, float] = {}
        self.error_count = 0
        self.start_time = time.time()
    
    def record_request(self, duration: float, operation: str = "request") -> None:
        """
        Record a request and its duration.
        
        Args:
            duration: Duration of the request in seconds
            operation: Type of operation (e.g., 'chat', 'transcribe', 'speak')
        """
        self.request_count += 1
        self.total_request_time += duration
        
        if operation in self.operation_counts:
            self.operation_counts[operation] += 1
            self.operation_times[operation] += duration
        else:
            self.operation_counts[operation] = 1
            self.operation_times[operation] = duration
        
        logger.debug(f"Recorded {operation} request: {duration:.4f}s")
    
    def record_error(self, operation: str = "request") -> None:
        """
        Record an error occurrence.
        
        Args:
            operation: Type of operation where error occurred
        """
        self.error_count += 1
        logger.debug(f"Recorded error for operation: {operation}")
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get global performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        uptime = time.time() - self.start_time
        avg_request_time = (self.total_request_time / self.request_count 
                           if self.request_count > 0 else 0.0)
        success_rate = ((self.request_count - self.error_count) / self.request_count 
                       if self.request_count > 0 else 0.0)
        
        stats = {
            "uptime": uptime,
            "total_requests": self.request_count,
            "total_request_time": self.total_request_time,
            "average_request_time": avg_request_time,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0.0
        }
        
        # Add operation-specific stats
        for operation, count in self.operation_counts.items():
            avg_time = self.operation_times[operation] / count if count > 0 else 0.0
            stats[f"{operation}_count"] = count
            stats[f"{operation}_avg_time"] = avg_time
            stats[f"{operation}_total_time"] = self.operation_times[operation]
        
        return stats
    
    def reset(self) -> None:
        """Reset all global statistics."""
        self.request_count = 0
        self.total_request_time = 0.0
        self.operation_counts.clear()
        self.operation_times.clear()
        self.error_count = 0
        self.start_time = time.time()
        logger.info("Reset global performance tracker")
    
    def log_stats(self) -> None:
        """Log current performance statistics."""
        stats = self.get_stats()
        
        logger.info("Global Performance Statistics:")
        logger.info(f"  Uptime: {stats['uptime']:.2f}s")
        logger.info(f"  Total Requests: {stats['total_requests']}")
        logger.info(f"  Average Request Time: {stats['average_request_time']:.4f}s")
        logger.info(f"  Success Rate: {stats['success_rate']:.2%}")
        logger.info(f"  Requests/Second: {stats['requests_per_second']:.2f}")
        
        if self.operation_counts:
            logger.info("  Operation Statistics:")
            for operation, count in self.operation_counts.items():
                avg_time = stats.get(f"{operation}_avg_time", 0.0)
                logger.info(f"    {operation}: {count} requests, {avg_time:.4f}s avg")


# Global instance
global_tracker = GlobalPerformanceTracker() 