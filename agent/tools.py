from typing import Dict, Any, List
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
import json
import sqlite3
import pandas as pd
from dowhy import CausalModel
from sklearn.preprocessing import LabelEncoder
from causaltune import CausalTune
from causaltune.data_utils import CausalityDataset
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import StandardScaler
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_numpy
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage


class CausalState(MessagesState):
    treatment: str
    outcome: str
    causal_graph: str
    variables: List[str]
    confounders: List[str]
    mediators: List[str]
    colliders: List[str]
    instruments: List[str]
    effect_modifiers: List[str]
    ate: float
    best_estimator: str
    best_score: float
    best_base_learner: str
    refutation_passed: bool
    refutation_results: List[Dict[str, str]]
    summary_result: str
    fixes_proposed: str
    error: str


VARIABLE_INFO_DICT = {
    "age": "Customer age",
    "income": "Customer income level",
    "education": "Education level of customer",
    "branch_visits": "Number of times the customer visited a physical branch",
    "channel_preference": "Preferred communication or service channels (e.g., online, phone, in-branch)",
    "customer_engagement": "Interactions, logins, responses to communication",
    "region_code": "Geographic region",
    "promotion_offer": "Whether the customer received a promotion",
    "activated_ib": "Whether the customer activated Internet Banking"
}

# Init LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")


def tool_calling(state: CausalState):
    VARIABLE_DESC = "The available variables are:\n" + "\n".join(
        [f"- {var}: {desc}" for var, desc in VARIABLE_INFO_DICT.items()]
    )
    
    # Extract original user question
    user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not user_msg:
        raise ValueError("Missing user question.")

    full_prompt = f"""
                    You are a causal inference assistant.
                    {VARIABLE_DESC}

                    User question:
                    "{user_msg.content.strip()}"

                    Please identify the treatment and outcome variables from the question, 
                    and use the `identify_causal_relationships` tool to identify causal relationships.
                  """
    
    return {"messages": [llm_with_tools.invoke([HumanMessage(content=full_prompt)])]}


def custom_tools_condition(state: CausalState) -> str:
    """Return the next node to execute."""
    messages = state["messages"]
    if not messages:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tool" 
    
    return "__end__"


@tool
def identify_causal_relationships(treatment: str, outcome: str):
    """
    Load data, run CausalNex to learn causal structure, return causal graph dot string.
    """
    with sqlite3.connect("data/bank.db") as conn:
        df = pd.read_sql_query("SELECT * FROM customer_data", conn)

    # Encode categorical features
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df_scaled = df_scaled.fillna(0)

    sm = from_pandas(df_scaled, w_threshold=0.2, max_iter=1000)
    sm.remove_edges_below_threshold(0.01)

    dot_str = "digraph {\n"
    for src, dst in sm.edges():
        dot_str += f"  {src} -> {dst};\n"
    dot_str += "}"
    
    # Also return dataframe columns for next nodes
    return {"treatment": treatment, "outcome": outcome, "causal_graph": dot_str, "variables": list(df.columns)}


def identify_causal_variables(state: CausalState):
    """
    Use LLM to classify variables into confounders, mediators, effect modifiers, colliders, instruments
    based on causal_graph and given treatment/outcome.
    """

    # Tìm ToolMessage mới nhất
    tool_msg = next(
        (msg for msg in reversed(state["messages"]) if msg.__class__.__name__ == "ToolMessage"),
        None
    )
    if not tool_msg:
        raise ValueError("No ToolMessage found in messages")

    result = json.loads(tool_msg.content)

    # Lấy các trường từ ToolMessage
    new_state = {}
    new_state['causal_graph'] = result["causal_graph"]
    new_state['variables'] = result["variables"]
    new_state['treatment'] = result["treatment"]
    new_state['outcome'] = result["outcome"]


    prompt = f"""
    You are a causal inference expert.

    Given the causal graph DOT:
    {new_state['causal_graph']}

    Treatment variable: {new_state['treatment']}
    Outcome variable: {new_state['outcome']}
    Variables: {new_state['variables']}

    Classify other variables as:
    - Confounders
    - Mediators
    - Effect Modifiers
    - Colliders
    - Instruments

    Return valid JSON with keys: confounders, mediators, effect_modifiers, colliders, instruments.
    """
    response = llm.invoke(prompt)
    try:
        var_roles = json.loads(response.content)
    except Exception:
        var_roles = {"error": "LLM failed to return valid JSON", "raw": response.content}

    new_state.update(var_roles)
    return {
        **new_state,
        "messages": state["messages"] + [AIMessage(content="Identified causal roles. " + json.dumps(var_roles))]
    }


def compute_ate(state: CausalState):
    """
    Run DoWhy causal model to identify and estimate treatment effect (ATE).
    """
    try:
        with sqlite3.connect("data/bank.db") as conn:
            df = pd.read_sql_query("SELECT * FROM customer_data", conn)
        # Preprocess categorical variables
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        model = CausalModel(
            data=df,
            treatment=state['treatment'],
            outcome=state['outcome'],
            common_causes=state['confounders'] or None,
            instruments=state['instruments'] or None,
            effect_modifiers=state['effect_modifiers'] or None,
        )

        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        ate = estimate.value

        return {
            "ate": ate,
            "messages": state["messages"] + [AIMessage(content=f"Computed ATE: {ate:.4f}")]
        }
    except Exception as e:
        return {"error": str(e), "messages": state["messages"]}


def tune_model(state: CausalState):
    """
    CausalTune: choose best estimator based on pre-defined list and data.
    """
    try:
        with sqlite3.connect("data/bank.db") as conn:
            df = pd.read_sql_query("SELECT * FROM customer_data", conn)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        estimators = ["S-learner", "T-learner", "X-learner"]
        # base_learners = ["random_forest", "neural_network"]

        cd = CausalityDataset(data=df, treatment=state['treatment'], outcomes=[state["outcome"]],
                            common_causes=state['confounders'])
        cd.preprocess_dataset()

        estimators = ["SLearner", "TLearner"]
        # base_learners = ["random_forest", "neural_network"]

        ct = CausalTune(
            estimator_list=estimators,
            metric="energy_distance",
            verbose=1,
            components_time_budget=10, # in seconds trial for each model
            outcome_model="auto",
        )

        # run causaltune
        ct.fit(data=cd, outcome=cd.outcomes[0])
        
        return {
            "best_estimator": ct.best_estimator,
            "best_score": ct.best_score,
            "messages": state["messages"] + [AIMessage(content=f"Best estimator: {ct.best_estimator}, score: {ct.best_score}")]
        }

    except Exception as e:
        return {"error": str(e), "messages": state["messages"]}


def refutation_test(state: CausalState):
    """
    Run DoWhy refutation tests on the best estimate.
    """
    try:
        with sqlite3.connect("data/bank.db") as conn:
            df = pd.read_sql_query("SELECT * FROM customer_data", conn)
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

        model = CausalModel(
            data=df,
            treatment=state['treatment'],
            outcome=state['outcome'],
            common_causes=state['confounders'] or None,
            instruments=state['instruments'] or None,
            effect_modifiers=state['effect_modifiers'] or None,
        )
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")

        # Run refutation methods
        refute_results = []
        refute_methods = [
            "placebo_treatment_refuter",
            "random_common_cause",
            "data_subset_refuter"
        ]
        for method in refute_methods:
            refute = model.refute_estimate(identified_estimand, estimate, method_name=method)
            refute_results.append({"method": method, "result": str(refute)})

        pass_test = all("fail" not in r["result"].lower() for r in refute_results)

        return {
            "refutation_results": refute_results,
            "refutation_passed": pass_test,
            "messages": state["messages"] + [AIMessage(content=f"Refutation passed: {pass_test}.\nRefutation results: {refute_results}")]
        }

    except Exception as e:
        return {"error": str(e), "messages": state["messages"]}


def summarize_result(state: CausalState):
    """
    Summarize the causal effect and refutation results with LLM, propose fixes if needed.
    """
    fixes = []
    if state.get('refutation_passed', "") == "":
        fixes = [
            "Check for omitted confounders or measurement errors.",
            "Collect more or better quality data.",
            "Re-examine and improve the causal graph.",
            "Consider better instrumental variables."
        ]

    prompt = f"""
    The estimated Average Treatment Effect (ATE) is {state['ate']:.4f}.

    Confounders: {state['confounders'] or 'None'}
    Effect Modifiers: {state['effect_modifiers'] or 'None'}
    Instrumental Variables: {state['instruments'] or 'None'}

    Refutation test results:
    {json.dumps(state['refutation_results'], indent=2)}

    Refutation tests passed: {state['refutation_passed']}

    If the refutation tests did not pass, suggest fixes:
    {fixes}

    Please write a clear, business-friendly summary explaining:
    - The strength and significance of the causal effect.
    - Confidence based on refutation tests.
    - Suggested next steps if refutation tests failed.
    - Do not mention instruction about the case refutation test fails when all the tests passed
    Avoid technical jargon.
    """
    response = llm.invoke(prompt)
    
    return {
        "summary_result": response.content.strip(),
        "fixes_proposed": fixes if not state['refutation_passed'] else [],
        "messages": state["messages"] + [AIMessage(content=response.content.strip())]
    }



# Bind tool to LLM
llm_with_tools = llm.bind_tools([identify_causal_relationships])