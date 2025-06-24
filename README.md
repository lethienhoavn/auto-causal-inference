# Auto Causal Inference for Banking

## Motivation

One of the most challenging aspects of **causal inference** is not running the estimation algorithm, but **correctly identifying the causal roles of variables** in the system — such as **confounders**, **mediators**, **colliders**, **effect modifiers**, and **instruments**.

This task typically requires domain expertise and experience, because:

* Simply adding **more variables** to the model does **not guarantee better causal estimates** — in fact, it can **bias** the results if colliders or mediators are adjusted incorrectly.
* Traditional approaches often rely on **manual DAG construction** and careful pre-analysis.

> ✅ **Auto Causal Inference** was created to solve this problem using **LLMs (Large Language Models)** — allowing users to specify only the **treatment** and **outcome**, and automatically infer variable roles and a suggested causal graph.

This enables:

* Faster experimentation with causal questions
* Automatically selecting the right confounding variables for the analysis
* Lower reliance on domain-specific manual DAGs
* More transparency and reproducibility in the inference process


## 🧠 Introduction

This project demonstrates an automated Causal Inference pipeline for banking use cases, where users only specify:

- a `treatment` variable
- an `outcome` variable

The app automatically:
- Uses a fixed list of variables (`age`, `income`, `education`, etc.)
- Calls an LLM to suggest causal structure (confounders, mediators, etc.)
- Returns a causal graph and executable DoWhy code

---

### 💼 Example use cases

| Scenario                                      | Treatment         | Outcome              |
|-----------------------------------------------|--------------------|-----------------------|
| Does promotion offer increase IB activation? | `promotion_offer` | `activated_ib`       |
| Do branch visits increase engagement?        | `branch_visits`   | `customer_engagement`|
| Does education level affect income?          | `education`       | `income`              |
| Does channel preference affect IB usage?     | `channel_preference` | `activated_ib`    |

### Lists of Variables for Analysis:

| Variable              | Description                                                                                                         |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| `age`                 | Customer age | 
| `income`              | Customer income level  | 
| `education`           | Education level of customer                                | 
| `branch_visits`       | Number of times the customer visited a physical branch in a time window                                            | 
| `channel_preference`  | Preferred communication or service channels (e.g., online, phone, in-branch)                                       | 
| `customer_engagement` | Composite metric reflecting interactions, logins, responses to comms, etc                                          |
| `region_code`         | Geographic region identifier                           | 
| `promotion_offer`     | Binary variable: whether the customer received a promotion                              | 
| `activated_ib`        | Binary outcome: whether the customer activated Internet Banking                 |



## Project Description

This project features two different agent architectures for running causal inference workflows:

- **LangGraph Agent:** Implements the analysis as a graph of tasks (nodes) executed synchronously or asynchronously, orchestrated in a single process.
- **MCP Agent:** Splits each task into independent MVP servers communicating via HTTP following the Model-Context-Protocol (MCP) pattern, enabling easy scaling and modular service deployment.


## Project Structure

```
auto_causal_inference/
├── agent/                 # LangGraph agent source code
│   ├── data/              # Sample data (bank.db)
│   ├── app.py             # Main entry point for LangGraph causal agent
│   ├── generate_data.py   # Data generation script for causal inference
│   ├── requirements.txt   # Dependencies for LangGraph agent
│   └── ...                # Other helper modules and notebooks
│
├── mcp_agent/             # MCP agent implementation
│   ├── data/              # Sample data (bank.db)
│   ├── server.py          # MCP causal inference server
│   ├── client.py          # MCP client to call the causal inference server
│   ├── requirements.txt   # Dependencies for MCP agent
│   └── ...                # Additional files
│
└── README.md              # This documentation file
````


## 📦 Requirements


- Python 3.10+
- Install dependencies:

```bash
pip install requirements.txt
````

## ▶️ How to Run

```bash
cd agent
python app.py

# OR
cd mcp_agent
python client.py
```

## 🧪 Input

```
User asks: "Does offering a promotion increase digital product activation ?"
```

## 📤 Output

```json
{
  "confounders": ["age", "income", "education"],
  "mediators": ["branch_visits"],
  "effect_modifiers": ["channel_preference"],
  "colliders": ["customer_engagement"],
  "instruments": ["region_code"],
  "causal_graph": "...DOT format...",
  "dowhy_code": "...Python code..."
}
```

### Causal Inference Relationships

```
age -> promotion_offer;
age -> activated_ib;
income -> promotion_offer;
income -> activated_ib;
education -> promotion_offer;
education -> activated_ib;

region_code -> promotion_offer;

promotion_offer -> branch_visits;
branch_visits -> activated_ib;

promotion_offer -> customer_engagement;
activated_ib -> customer_engagement;

channel_preference -> activated_ib;
promotion_offer -> activated_ib
```

### DoWhy code

```python
import dowhy
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment='promotion_offer',
    outcome='activated_ib',
    common_causes=['age', 'income', 'education'],
    instruments=['region_code'],
    mediators=['branch_visits']
)

identified_model = model.identify_effect()
estimate = model.estimate_effect(identified_model, method_name='backdoor.propensity_score_matching')
print(estimate)
```

### Summary of Variable Roles:

| Role                | Variable                     | Why it's assigned this role                                      |
| ------------------- | ---------------------------- | ---------------------------------------------------------------- |
| **Confounder**      | `age`, `income`, `education` | Affect both the chance of receiving promotions and IB usage.     |
| **Mediator**        | `branch_visits`              | A step in the causal path: promotion → visit → IB activation.    |
| **Effect Modifier** | `channel_preference`         | Alters the strength of the effect of promotion on IB activation. |
| **Collider**        | `customer_engagement`        | Affected by both promotion and IB usage; should not be adjusted. |
| **Instrument**      | `region_code`                | Randomized promotion assignment at the regional level.           |


### Result Analysis:

1. There is a causal effect between offering promotions and activating internet banking services, with a 15% increase of activating internet banking if we open the promotion for everybody. This shows a strong positive impact of the promotion offer on activation.

2. Factors like age, income, education level could have influenced both the decision to offer promotions and the likelihood of activating internet banking services. These factors may have affected the outcome regardless of the promotion offer.
