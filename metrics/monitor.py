import time
import torch
import pandas as pd
import psutil
import os
import numpy as np
from datetime import datetime

class PerformanceMonitor:
    """
    The Universal Ruler.
    Tracks Latency, VRAM, Entropy, and Accuracy for any model architecture.
    """
    def __init__(self, experiment_name, output_dir="results/logs"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.os_process = psutil.Process(os.getpid())
        
        # storage
        self.records = []
        self.start_time = None
        
        # Ensure dir exists
        os.makedirs(output_dir, exist_ok=True)

    def start_frame(self):
        """Call this before `model.predict()`"""
        self.start_time = time.perf_counter()
        # Reset GPU peak memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def end_frame(self, true_label, pred_label, entropy, adapter_name):
        """Call this after `model.predict()`"""
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        
        # Memory Usage
        vram_mb = 0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        elif torch.backends.mps.is_available():
            vram_mb = self.os_process.memory_info().rss / (1024 ** 2) # Approximation for MPS
            
        record = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_name,
            "adapter_active": adapter_name,
            "latency_ms": round(latency_ms, 2),
            "vram_mb": round(vram_mb, 2),
            "entropy": round(float(entropy), 4),
            "ground_truth": true_label,
            "prediction": pred_label,
            "correct": 1 if true_label == pred_label else 0
        }
        self.records.append(record)

    def save_report(self):
        """Dumps the session to CSV"""
        df = pd.DataFrame(self.records)
        filename = f"{self.output_dir}/{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        
        # Print Summary
        print(f"\n--- REPORT: {self.experiment_name} ---")
        print(f"AVG Latency: {df['latency_ms'].mean():.2f} ms")
        print(f"Accuracy:    {df['correct'].mean() * 100:.1f}%")
        print(f"AVG Entropy: {df['entropy'].mean():.4f}")
        print(f"Log saved to: {filename}")
        return filename
