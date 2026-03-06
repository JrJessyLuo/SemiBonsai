# SemiBonsai

SemiBonsai is a semi-structured table QA framework with four main components:

- **Structurer**: builds a structural model for semi-structured tables (via VLM).
- **Resolver**: resolves question uncertainty.
- **Reasoner**: generates and executes query plans and reasoning programs over the structural model.
- **Router**: bandit LLM router under budget constraints.

---

## 1) Code Structure

```
SemiBonsai/
├─ bash/
│  └─ environment.sh                  # Set environment variables (API keys / provider)
├─ codes/
│  ├─ table_structurer/               # Structurer (VLM-based table structuring)
│  ├─ uncertainty_resolver/           # Resolver (uncertainty detection and question rewriting)
│  ├─ reasoner/                       # Reasoner (query planning + execution)
│  ├─ router/                         # Router (LLM routing under budgets)
│  └─ utils/                          # Helper functions (evaluation, IO, prompts, etc.)
├─ datasets.zip/
│  ├─ annotations/                    # Statistics for complexity factors of tables/questions
│  ├─ complex_subsets/                # Curated complex subsets
│  └─ test_dataset/                   # Example benchmark folder
│     └─ data/
│        ├─ single_tab_qa.jsonl       # QA file
│        └─ table/                    # Table files
├─ result/
│  └─ ...                             # Intermediate outputs + final outputs for each component
└─ config.yaml                        # Dataset path configuration + router configuration
```

---

## 2) Installation

### 2.1 Install dependencies
```bash
pip install -r requirements.txt
```

### 2.2 Set environment variables
You need to set the required variables such as loading API keys before running VLM/LLM modules.

```bash
source bash/environment.sh
```

### 2.3 Set PYTHONPATH
```bash
# Replace <SemiBonsai_root> with the absolute path to your SemiBonsai repo
export PYTHONPATH="<SemiBonsai_root>/codes"
```

---

## 3) Configuration (config.yaml)

Only **two fields must be configured by the user**:

- `base_folder`: where your datasets live (this repo’s `datasets/`)
- `out_folder`: where outputs/results should be written (this repo’s `result/`)

---

## 4) How to Run SemiBonsai

Below is the standard pipeline: Structurer → Resolver → Reasoner → Evaluation. You can use our benchmarks by unzip the datasets.zip.

### 4.1 Structurer 
```bash
cd codes/table_structurer

# header identification
python vlm_identification.py --dataset test_dataset

# multiple subtables identification
python vlm_identification.py --dataset test_dataset --multiple

# convert into our structural model
python convert_table_structural_model.py --dataset test_dataset
```

### 4.2 Resolver 
```bash
cd ../uncertainty_resolver
python question_rewriting.py --dataset test_dataset
```

### 4.3 Reasoner 
```bash
cd ../reasoner
python query_plan.py --dataset test_dataset
```

### 4.4 Evaluate answer accuracy
```bash
cd ../utils
python evaluate_utils.py
```

---

## 5) Router: bandit LLM routing

Before running the router, you should prepare: see the examples using the file under datasets/hitab_num/input/
- train instances file: 
- eval instances file

Then place them into the paths specified by `llm_routing.datasets.<dataset>.train_path` and `eval_path` in `config.yaml`.

Run:
```bash
cd codes/router
python llm_routing.py --dataset test_dataset --budgets 0.1 0.2 0.3
```

Note: Choose the values for --budgets based on the budget scale and cost distribution of your dataset.

---