# SAGE-Loop (under review)

Closed-loop, LLM-driven AutoML that learns to **trial–and–correct** and **use models intelligently** on tabular tasks.  
> At this stage, the repository exposes **ensemble (stacking / voting / bagging)** code only. Other modules will be released in phases depending on the review timeline.

---

## ⚠️ Usage & Rights (Important)

**Under review — No open-source license granted at this stage. ALL RIGHTS RESERVED.**

- The repository is shared **for paper evaluation and artifact verification only**.
- **No permission** is granted to **copy, modify, distribute, or create derivative works**.
- For broader usage, please **contact the maintainer to obtain written permission**.
- A formal open-source license (e.g., Apache-2.0 / BSD-3-Clause / MIT) may be added **after review**.

---

## What’s Inside (current)
- ✅ Ensemble component (stacking / voting / bagging; level-2 learner variants)
- ⏳ Closed-loop pipeline (feature generation, multi-round model synthesis, clustering consensus) — **to be staged post-review**

---

## Quick Start (ensemble-only)

**Python ≥ 3.9 (3.10 recommended)**

```bash
# (optional) create and activate a virtual environment
python -m venv .venv && . .venv/bin/activate

# minimal dependencies for the ensemble component
pip install -U numpy pandas scikit-learn xgboost lightgbm
```

Then follow the instructions under `SAGE-Loop/` to run examples (use your local datasets).

---

## Data
This repository **does not** ship any datasets. Prepare local copies (e.g., FinBench, UCI/Kaggle) and follow their original licenses.

---

## Minimal Layout
```
SAGE-Loop/      # public ensemble module
README.md
(other closed-loop modules will be opened in stages)
```

---

## Citation
If you use SAGE-Loop or its ideas, please cite:
```bibtex
@misc{Gu2025SAGELoop,
  title        = {SAGE-Loop: Closed-Loop LLM-Driven AutoML that Learns to Trial-and-Correct and Use Models Intelligently},
  author       = {Gu, Junquan and Yu, Hang and Luo, Xiangfeng and Liu, Zhengyang and Cui, Shibo and Wu, Jia},
  year         = {2025},
  howpublished = {GitHub repository and manuscript under review},
  url          = {https://github.com/sethGu/SAGE-Loop}
}
```

---

## Contact
Maintainer: Junquan Gu · gujunquan@shu.edu.cn
