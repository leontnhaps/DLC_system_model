#!/usr/bin/env python3
"""
PV (PhotoVoltaic) Voltage/Current Monitor
Collects real-time voltage, current, and power data from Arduino via serial
"""

import serial
import threading
import time
from collections import deque
from typing import Optional, Tuple, List
import queue
from datetime import datetime
import csv
import pathlib

class PVMonitor:
    """Monitor for real-time voltage/current data from Arduino"""
    
    def __init__(self, max_history: int = 100, output_dir: Optional[str] = None):
        """
        Initialize PV Monitor
        
        Args:
            max_history: Maximum number of data points to keep in history
        """
        self.port: Optional[str] = None
        self.baud_rate: int = 9600
        self.serial_conn: Optional[serial.Serial] = None
        self.output_dir = output_dir  # 추가
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Data storage (thread-safe)
        self.max_history = max_history
        self._voltage_history = deque(maxlen=max_history)
        self._current_history = deque(maxlen=max_history)
        self._power_history = deque(maxlen=max_history)
        self._time_history = deque(maxlen=max_history)
        
        # Recording control
        self._recording = False
        # 라인 42 근처에 추가
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None
        
        # Latest values
        self._latest_voltage: float = 0.0
        self._latest_current: float = 0.0
        self._latest_power: float = 0.0
        
        # Error tracking
        self._error_queue = queue.Queue(maxsize=10)
        self._last_error: Optional[str] = None
    
    def start_monitoring(self, port: str, baud_rate: int = 9600) -> bool:
        """
        Start monitoring Arduino serial data
        
        Args:
            port: Serial port name (e.g., 'COM8')
            baud_rate: Baud rate (default: 9600)
            
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            self._set_error("Already running")
            return False
        
        try:
            # Open serial connection
            self.serial_conn = serial.Serial(port, baud_rate, timeout=1)
            self.port = port
            self.baud_rate = baud_rate
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Start reading thread
            self._running = True
            self._recording = False  # Default to not recording
            self._thread = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            
            # 라인 88-90 수정
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"pv_data_{timestamp}.csv"
            # output_dir이 있으면 그 안에 저장
            if self.output_dir:
                import pathlib
                self._csv_path = pathlib.Path(self.output_dir) / csv_filename
            else:
                self._csv_path = csv_filename
            self._csv_file = open(self._csv_path, 'w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(['Time(s)', 'Voltage(V)', 'Current(mA)', 'Power(mW)'])
            
            print(f"[PVMonitor] Started monitoring on {port}")
            return True
            
        except serial.SerialException as e:
            self._set_error(f"포트 {port} 열기 실패: {str(e)}")
            return False
        except Exception as e:
            self._set_error(f"시작 오류: {str(e)}")
            return False
    
    def stop_monitoring(self):
        """Stop monitoring"""
        if not self._running:
            return
        
        self._running = False
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.clear_history()
        if self._csv_file:
            self._csv_file.close()
            print(f"[PVMonitor] CSV saved: {self._csv_path}")
        
        # Close serial connection
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        
        print("[PVMonitor] Stopped monitoring")
    
    def _read_loop(self):
        """Main reading loop (runs in separate thread)"""
        start_time = time.time()
        
        while self._running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    # Read one line
                    line = self.serial_conn.readline().decode('utf-8').strip()
                    
                    # Parse data (format: "voltage,current,power")
                    data = line.split(',')
                    
                    if len(data) == 3:
                        try:
                            voltage = float(data[0])
                            current = float(data[1])
                            power = float(data[2])
                            
                            # Store data (thread-safe)
                            elapsed_time = time.time() - start_time
                            with self._lock:
                                # Update latest always
                                self._latest_voltage = voltage
                                self._latest_current = current
                                self._latest_power = power

                                # Append to history ONLY if recording is active
                                if self._recording:
                                    self._voltage_history.append(voltage)
                                    self._current_history.append(current)
                                    self._power_history.append(power)
                                    self._time_history.append(elapsed_time)
                                    if self._csv_writer:
                                        self._csv_writer.writerow([elapsed_time, voltage, current, power])
                                
                        except ValueError:
                            # Invalid number format, skip
                            pass
                
                # Small delay to prevent CPU overuse
                time.sleep(0.01)
                
            except Exception as e:
                if self._running:  # Only log if still supposed to be running
                    self._set_error(f"읽기 오류: {str(e)}")
                    time.sleep(0.5)  # Wait before retry
    
    def set_recording(self, enable: bool):
        """Set execution state of recording history"""
        with self._lock:
            self._recording = enable

    def get_latest_data(self) -> Tuple[float, float, float]:
        """
        Get the latest voltage, current, power values
        
        Returns:
            (voltage, current, power) tuple
        """
        with self._lock:
            return (self._latest_voltage, self._latest_current, self._latest_power)
    
    def get_data_history(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Get historical data for graphing
        
        Returns:
            (time_list, voltage_list, current_list, power_list) tuple
        """
        with self._lock:
            return (
                list(self._time_history),
                list(self._voltage_history),
                list(self._current_history),
                list(self._power_history)
            )
    
    def clear_history(self):
        """Clear all historical data"""
        with self._lock:
            self._voltage_history.clear()
            self._current_history.clear()
            self._power_history.clear()
            self._time_history.clear()
    
    def is_running(self) -> bool:
        """Check if monitoring is active"""
        return self._running
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message"""
        return self._last_error
    
    def _set_error(self, error_msg: str):
        """Set error message (thread-safe)"""
        self._last_error = error_msg
        try:
            self._error_queue.put_nowait(error_msg)
        except queue.Full:
            pass  # Queue full, ignore
        print(f"[PVMonitor ERROR] {error_msg}")


def test_monitor():
    """Test function (run standalone)"""
    monitor = PVMonitor()
    
    # Start monitoring
    if monitor.start_monitoring('COM8'):
        print("Monitoring started. Press Ctrl+C to stop.")
        try:
            while True:
                voltage, current, power = monitor.get_latest_data()
                print(f"☀️ V: {voltage:.2f}V | I: {current:.2f}mA | P: {power:.2f}mW")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            monitor.stop_monitoring()
    else:
        print(f"Failed to start: {monitor.get_last_error()}")


if __name__ == "__main__":
    test_monitor()
