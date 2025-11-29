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

## 🚀 Quick Start
### Create Conda Environment and Install Dependencies
```bush
conda create -n sage-loop python=3.11.12
conda activate sage-loop
cd SAGE-Loop
pip install -r requirements.txt
```
### Run SAGE-Loop
```bush
cd SAGE-Loop
export OPENAI_BASE_URL="openai base_url"
export OPENAI_API_KEY="your aip_key"
./run_SAGE_Loop.sh
```
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
