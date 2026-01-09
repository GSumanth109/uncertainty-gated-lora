# AGENT INSTRUCTIONS (STRICT)

**📋 Task Management:**
Before starting work, ALWAYS read `TASKS.md` to see what is pending. 
If you complete a task, mark it with `[x]` in your commit.

## 🎯 Current Mission: Phase 1 - Reproduce the Problem
**Goal:** Quantify the "Domain Shift Degradation" of a static ResNet-50 model.
**Success Metric:** Generate a report showing the accuracy drop between "Sunny" (Source Domain) and "Rain/Night" (Target Domains).

**We are NOT building the solution yet.** Do not implement LoRA, Gating, or Adapters.
Focus ONLY on:
1. Setting up the BDD100K data loader.
2. Running a standard Pre-trained ResNet-50 on Sunny vs. Rainy/Night images.
3. Logging the accuracy drop and "Entropy" (Model Confusion) levels.

## Project Context
This project investigates "Uncertainty-Gated LoRA." However, currently, we are establishing the **Baseline**.
We hypothesize that standard models fail in non-stationary environments. We need hard data to prove this.

## Technical Constraints
- **Model:** `ResNetForImageClassification` (Pre-trained on ImageNet).
- **Metric:** Accuracy (%) and Shannon Entropy.
- **Hardware:** CPU/MPS/CUDA (Auto-detect).

## Project Change Log
(Do not edit below this line manually. The system updates this automatically.)
-------------------------------------------------------------------------------
- **[2026-01-09 05:06] Sumanth Gopisetty:** ci: Add auto-scribe workflow to track history
