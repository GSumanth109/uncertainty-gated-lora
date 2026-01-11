import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add root to path so we can import 'core'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.factory import AdapterFactory

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    # 1. Build Model via Factory
    model, cfg = AdapterFactory.create_model(args.config)
    device = cfg['training']['device'] if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 2. Setup (Mock) Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    print(f"🚀 [Trainer] System Ready on {device.upper()}")
    print("   [Test] Running 1 forward pass to verify architecture...")

    # 3. Smoke Test (One fake batch)
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0, 1]).to(device)
    
    output = model(dummy_input).logits
    loss = criterion(output, dummy_labels)
    loss.backward()
    optimizer.step()
    
    print("✅ [Trainer] Forward/Backward pass successful. Architecture is stable.")

if __name__ == "__main__":
    train()
