# SAGE-Loop (under review)

Closed-loop, LLM-driven AutoML that learns to **trial–and–correct** and **use models intelligently** on tabular tasks.  
> At this stage, the repository exposes **ensemble (stacking / voting / bagging)** code only. Other modules will be released in phases depending on the review timeline.

---
## 🧩 Framework Overview
<p align="center">
  <img src="./tests/assets/method.png">
</p>
### 1. Feature Generation Module
The feature generation stage leverages an LLM to create new candidate features using dataset metadata, column descriptions, and partial samples drawn **only from the training set**. The LLM receives a structured prompt containing:
- Dataset schema and inferred datatypes
- Downstream task description (classification, regression, clustering)
- A small number of sampled rows from the **training split only**

Based on this prompt, the LLM synthesizes Python code that constructs new feature columns (e.g., ratios, interactions, aggregations). The system automatically executes the generated code, validates correctness, filters invalid outputs, and evaluates the usefulness of each feature using:
- Mutual information (classification)
- F-test (regression)
- Variance / structural scores (clustering)

Top-K selected features are passed onward into the modeling stage.

---

### 2. Model Generation and Closed-loop Optimization
The model generation stage uses an LLM to propose candidate models and training code. The prompt includes:
- Task type and target column name
- Partial training samples
- Feature list (original + engineered)

A **closed-loop refinement mechanism** is adopted:
1. **Code Generation** – LLM produces candidate model code.
2. **Execution** – The system runs the model code inside a controlled environment.
3. **Error Detection** – If execution fails, stack traces are captured.
4. **Error Repair Prompt** – The failure logs are returned to the LLM to fix the code.
5. **Performance Feedback Prompt** – Validation metrics are summarized and sent back, encouraging performance-driven refinement.

This loop repeats for several rounds to obtain a pool of diverse, validated candidate models.

---

### 3. Ensemble Learning Module
The final stage aggregates candidate models using stacking, bagging, or voting. The system contains:
- Automatic model-level filtering to remove degenerate or consistently underperforming models
- LLM-assisted selection of appropriate ensemble strategy
- Optional LLM-synthesized stacker (level-2 learner)
- Consensus-based aggregation for clustering tasks (co-association matrix + filtering)

The ensemble uses **only validation predictions from the training split** to ensure independence from the test data.

---

### Data Splitting and Leakage Prevention
SAGE-Loop enforces strict data separation:
- The dataset is split into **train**, **validation**, and **test** before any LLM interaction.
- All LLM prompts involving data include **only training samples**.
- Validation data is used strictly for performance scoring.
- **Test data is never exposed to the LLM**, and is used *only once* at the final ensemble evaluation stage.

This ensures that no train–test leakage occurs throughout feature generation, model generation, or ensemble construction.

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


## ⚠️ Usage & Rights (Important)

**Under review — No open-source license granted at this stage. ALL RIGHTS RESERVED.**

- The repository is shared **for paper evaluation and artifact verification only**.
- **No permission** is granted to **copy, modify, distribute, or create derivative works**.
- For broader usage, please **contact the maintainer to obtain written permission**.
- A formal open-source license (e.g., Apache-2.0 / BSD-3-Clause / MIT) may be added **after review**.

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
