import pickle
from google import genai
from google.genai import types
import os
import json
import pandas as pd

# ------------------------
# Load Artifact
def load_artifact(path_to_artifact):
    """
    Loads a serialized artifact (e.g., pipeline, model, encoder, scaler) from a pickle file.

    Args:
        path_to_artifact (str): File path to the pickle file.

    Returns:
        Any: The loaded Python object from the pickle file.
    """
    with open(path_to_artifact, 'rb') as f:
        artifact = pickle.load(f)
    return artifact
 
# ------------------------
# LLM for Feature Importance
google_credentials = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

def parse_llm_output(llm_output):
    """
    Parse LLM output containing JSON to a Python dictionary.
    """
    try:
        if llm_output.startswith('```json'):
            llm_output = llm_output[7:]
        if llm_output.endswith('```'):
            llm_output = llm_output[:-3]
        parsed_data = json.loads(llm_output.strip())
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}")

def llm_explain_feature_importance(feature_importance_str: str):
    """
    Generate textual explanation of feature importance for a single prediction.

    Args:
        feature_importance_str (str): Feature importance table as string

    Returns:
        dict: {"explanation": "..."}
    """
    client = genai.Client(api_key=google_credentials)

    prompt_text = types.Part.from_text(text=f"""
    You are an AI specialized in explaining machine learning model predictions in a banking context. Given the following feature importance for a single prediction, your taks is to
    provide a clear and concise explanation of how the features influenced the likelihood that the customer will ACCEPT a long-term deposit offer. Focus on the features with higher importance.

    The model was trained on historical campaign data, which includes personal information (age, job, marital status, education), financial information (balance, housing loan, personal loan, credit default), and campaign interaction information (contact type, day, month, previous contacts, previous campaign outcome, duration of last contact).

    ## INPUT LIST
    * Feature importance:
    {feature_importance_str}

    ## EXAMPLES
    # Example format of input:
    # Feature | Importance
    # age      0.12
    # balance  0.34
    # duration 0.55


    ## JSON OUTPUT FORMAT
    Return a JSON object:
    {{
      "explanation": "Text explanation of how the features influenced the prediction."
    }}

    ONLY return JSON. Do not add extra text or markdown.

    ## ATTENTION
    * ALWAYS respond professionally.
    * Do NOT include extra text outside the JSON.
    * If the effect of a feature is negligible, still provide a short explanation reflecting its minor influence.
    * Respond ONLY with the JSON object.
    """)
    
    contents = [types.Content(role="user", parts=[prompt_text])]
    
    generation_config = types.GenerateContentConfig(
        temperature=0,
        top_p=0.9,
        max_output_tokens=800,
        response_modalities=['TEXT']
    )
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=contents,
            config=generation_config
        )
        llm_output = response.candidates[0].content.parts[0].text
        parsed_output = parse_llm_output(llm_output)
        return parsed_output
    except Exception as e:
        return {"explanation": f"ERROR: {str(e)}"}