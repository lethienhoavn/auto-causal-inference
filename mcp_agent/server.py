from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP
from langchain_core.tools import tool
from typing import Dict
import sqlite3
import pandas as pd
import json
from dowhy import CausalModel
from langchain_openai import ChatOpenAI

VARIABLE_INFO_DICT = {
    "age": "Customer age",
    "income": "Customer income level",
    "education": "Education level of customer",
    "branch_visits": "Number of times the customer visited a physical branch",
    "channel_preference": "Preferred communication or service channels",
    "customer_engagement": "Interactions, logins, responses to communication",
    "region_code": "Geographic region",
    "promotion_offer": "Whether the customer received a promotion",
    "activated_ib": "Whether the customer activated Internet Banking",
}

llm = ChatOpenAI(model="gpt-3.5-turbo")

mcp = FastMCP("CausalInference")

@mcp.tool()
def auto_causal_inference(
    treatment: str,
    outcome: str
) -> Dict[str, str]:
    """
    LLM suggests variable roles and DoWhy code, then code is executed on real SQLite data.

    Args:
        treatment: Treatment variable
        outcome: Outcome variable

    Returns:
        Dict with causal role analysis, causal graph (dot), DoWhy code and result analysis
    """

    prompt = f"""
        You are a causal inference expert.
        Given the following variables: {list(VARIABLE_INFO_DICT.keys())}
        Treatment: {treatment}, Outcome: {outcome}

        Classify all variables into:
        - Treatment
        - Outcome
        - Confounders
        - Mediators
        - Effect Modifiers
        - Colliders
        - Instruments

        Draw DOT causal graph.
        Write DoWhy code.

        Return valid JSON with keys: treatment, outcome, confounders, mediators, effect_modifiers, colliders, instruments, causal_graph, dowhy_code
    """

    response = llm.invoke(prompt)
    try:
        result = json.loads(response.content)
    except Exception:
        return {"error": "Invalid JSON from LLM", "raw": response.content}

    try:
        with sqlite3.connect("data/bank.db") as conn:
            df = pd.read_sql_query("SELECT * FROM customer_data", conn)

        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=result.get("confounders", []),
            instruments=result.get("instruments", []),
            effect_modifiers=result.get("effect_modifiers", []),
        )

        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        ate = estimate.value
        result["causal_effect"] = ate

        summary_prompt = f"""
            The ATE of `{treatment}` on `{outcome}` is {ate:.4f}.
            Confounders: {', '.join(result.get('confounders', []))}
            Effect Modifiers: {', '.join(result.get('effect_modifiers', []))}
            Instruments: {', '.join(result.get('instruments', []))}

            Write a plain language summary (3-4 sentences) about the causal effect and strength.
        """
        summary = llm.invoke(summary_prompt)
        result["causal_summary"] = summary.content.strip()

    except Exception as e:
        result["execution_error"] = str(e)

    return result

if __name__ == "__main__":
    mcp.run()
    print("Start server successfully !")
