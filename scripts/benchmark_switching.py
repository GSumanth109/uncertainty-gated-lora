import sys
import os
import time
import torch
import numpy as np

# Fix path to allow importing from 'core'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.backbone import AdaptiveVisionModel

def run_benchmark():
    # 1. Initialize
    vision_system = AdaptiveVisionModel()
    
    # 2. Create Dummy "Experts"
    print("\n--- CREATING EXPERTS ---")
    vision_system.add_adapter("sunny")
    vision_system.add_adapter("rainy")
    vision_system.add_adapter("night")
    print("Adapters loaded into VRAM.")

    # 3. Create Dummy Input (Batch size 1)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. Benchmark Loop
    print("\n--- STARTING LATENCY TEST (100 Switches) ---")
    start_time = time.time()
    iterations = 100
    
    # Pre-warm
    vision_system.switch_adapter("sunny")
    vision_system.predict(dummy_input)

    for i in range(iterations):
        # Simulate Logic: Switch every frame to stress-test
        if i % 3 == 0:
            target = "sunny"
        elif i % 3 == 1:
            target = "rainy"
        else:
            target = "night"
            
        vision_system.switch_adapter(target)
        _ = vision_system.predict(dummy_input)

    end_time = time.time()
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000  # Convert to ms

    print(f"\nRESULTS:")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Switch+Inference Latency: {avg_latency:.2f} ms")
    print(f"Projected Max FPS: {1000/avg_latency:.1f} FPS")
    
    if avg_latency < 33:
        print("\n✅ SUCCESS: System is Real-Time Capable (>30 FPS)")
    else:
        print("\n⚠️ WARNING: Latency is high. Optimization needed.")

if __name__ == "__main__":
    run_benchmark()
