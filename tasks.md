# 📋 Project Quest Board

**Instructions for Agents:**
1. Read this file to find an unclaimed task (marked `[ ]`).
2. If the user wants to work on it, edit this file to write their name next to "Assigned:".
3. When the work is committed, change `[ ]` to `[x]`.

---

## 🚀 Phase 1: The Baseline (Current Mission)
**Goal:** Prove that ResNet-50 fails in Rain/Night.

### 📦 Data Collection
- [ ] **Task 1.1:** Find 10 "Sunny/Clear" driving images.
    - *Action:* Save to `data/test_sunny/`
    - *Assigned:* - [ ] **Task 1.2:** Find 10 "Rainy/Wet" driving images.
    - *Action:* Save to `data/test_rain/`
    - *Assigned:* - [ ] **Task 1.3:** Find 10 "Night/Dark" driving images.
    - *Action:* Save to `data/test_night/`
    - *Assigned:* ### 📊 Analysis
- [ ] **Task 1.4:** Run the Baseline Script.
    - *Action:* Run `python scripts/establish_baseline.py`
    - *Output:* Paste the "Entropy Score" results below.
    - *Assigned:* **Phase 1 Results:**
(Paste results here after Task 1.4 is done)

---

## 🔒 Phase 2: Building Experts (Locked)
*(Do not start these tasks until Phase 1 is complete)*

- [ ] **Task 2.1:** Train "Sunny" LoRA Adapter.
- [ ] **Task 2.2:** Train "Rain" LoRA Adapter.
- [ ] **Task 2.3:** Train "Night" LoRA Adapter.

### Phase 1 Verification Logs
| Domain | Entropy | Status |
|--------|---------|--------|
| Sunny  | 4.1044  | CONFUSED (Expected) |
| Rain   | 1.6730  | CONFUSED |
| Night  | 0.1744  | Confident (False Positive) |
- [ ] **Task 1.5:** Generate Visualization & LaTeX Report.
    - *Action:* Create `report/reproducer.py` and `report/gen_latex.py`.
