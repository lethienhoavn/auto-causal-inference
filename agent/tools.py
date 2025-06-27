from typing import List, Dict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessagesState
import json
import sqlite3
from dowhy import CausalModel
from sklearn.preprocessing import LabelEncoder
import pandas as pd


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


@tool
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
    variables = list(VARIABLE_INFO_DICT.keys())

    # Step 1: Prompt LLM
    prompt = f"""
                You are a causal inference expert.

                Given variables: {variables}
                Treatment: {treatment}
                Outcome: {outcome}

                Classify other variables as:
                - Treatment
                - Outcome
                - Confounders
                - Mediators
                - Effect Modifiers
                - Colliders
                - Instruments

                Then:
                - Draw DOT causal graph
                - Write DoWhy code to estimate effect

                Return the result in JSON format with keys: treatment, outcome, confounders, mediators, effect_modifiers, colliders, instruments, causal_graph, dowhy_code.
                All in valid JSON, properly escaped and parsable. Avoid multiline strings or escape characters like \\n, \\"..."
            """
    response = llm.invoke(prompt)

    try:
        result = json.loads(response.content)
    except Exception:
        return {"error": "LLM failed to return valid JSON", "raw": response.content}

    # Step 2: Execute DoWhy code
    try:
        # Load real data
        with sqlite3.connect("data/bank.db") as conn:
            df = pd.read_sql_query("SELECT * FROM customer_data", conn)

        # Convert categorical columns to strings then encode
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])    

        # Prepare model
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=result.get("confounders", []) or None,
            instruments=result.get("instruments", []) or None,
            effect_modifiers=result.get("effect_modifiers", []) or None,
        )

        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        ate = estimate.value

        result["causal_effect"] = ate

        # Step 3: LLM generates natural language explanation
        summary_prompt = f"""
                            You are a causal inference analyst.

                            The estimated Average Treatment Effect (ATE) of `{treatment}` on `{outcome}` is {ate:.4f}.

                            Additional information:
                            - Confounders: {', '.join(result.get("confounders", []) or ['None'])}
                            - Effect Modifiers: {', '.join(result.get("effect_modifiers", []) or ['None'])}
                            - Instrumental Variables: {', '.join(result.get("instruments", []) or ['None'])}

                            Please write a short, clear explanation (3-4 sentences) that:
                            1. States whether there is a causal effect and how strong it is, says in %. Small ATE (absolute value < 0.01) means no causal effect between treatment and outcome.
                               Remember that ATE is the % increase of outcome when everybody is applied the treatment.
                            2. Mentions any confounders that may have influenced both treatment and outcome.
                            3. Notes any effect modifiers or instruments that might have impacted estimation.

                            Avoid technical jargon; use plain, business-friendly language, do not use term confounders, effect modifiers, instrumental variables, treatment, outcome, ATE.
                        """

        summary_response = llm.invoke(summary_prompt)
        result["causal_summary"] = summary_response.content.strip()

    except Exception as e:
        result["execution_error"] = str(e)

    return result
