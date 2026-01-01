"""
Prompt templates for ML explanations

Phase 3.4.1: Design prompts for each ML model type
These templates will be used to generate human-readable maintenance explanations
from ML predictions + RAG context
"""

# System prompt - Sets the role and guidelines for the LLM
SYSTEM_PROMPT = """You are an expert industrial maintenance engineer with 20+ years of experience. 
Your role is to explain machine learning predictions to maintenance technicians in clear, actionable language.

Guidelines:
- Use simple, non-technical language
- Focus on practical actions
- Include safety considerations
- Estimate cost and downtime
- Be concise but thorough
"""


# Template for Failure Classification Model explanations
FAILURE_CLASSIFICATION_PROMPT = """A predictive maintenance model has analyzed {machine_id} and detected:

PREDICTION: {failure_probability:.1%} probability of {failure_type}
CONFIDENCE: {confidence:.1%}

SENSOR READINGS:
{sensor_readings}

RETRIEVED CONTEXT:
{rag_context}

Provide a maintenance explanation covering:
1. What this prediction means
2. Why the model flagged this (which sensors are abnormal)
3. Immediate actions to take
4. Expected cost and downtime
5. Safety precautions

Keep response under 200 words."""


# Template for RUL Regression Model explanations
RUL_REGRESSION_PROMPT = """A predictive maintenance model estimates {machine_id} has:

REMAINING USEFUL LIFE: {rul_hours:.0f} hours ({rul_days:.1f} days)
CONFIDENCE: {confidence:.1%}

SENSOR TRENDS:
{sensor_readings}

RETRIEVED CONTEXT:
{rag_context}

Explain:
1. What RUL means for this machine
2. Key factors driving this estimate
3. Maintenance scheduling recommendations
4. What to monitor closely
5. Risk if maintenance is delayed

Keep response under 200 words."""


# Template for Anomaly Detection Model explanations
ANOMALY_DETECTION_PROMPT = """Anomaly detection flagged unusual behavior in {machine_id}:

ANOMALY SCORE: {anomaly_score:.2f} (threshold: 0.5)
DETECTED BY: {detection_method}

ABNORMAL SENSORS:
{abnormal_sensors}

RETRIEVED CONTEXT:
{rag_context}

Explain:
1. What the anomaly indicates
2. Which sensors are unusual and why
3. Potential causes
4. Immediate investigation steps
5. Urgency level (low/medium/high)

Keep response under 200 words."""


# Template for Time-Series Forecast Model explanations
TIMESERIES_FORECAST_PROMPT = """Time-series forecast for {machine_id} predicts:

FORECAST (Next 24h):
{forecast_summary}

PREDICTION CONFIDENCE: {confidence:.1%}

RETRIEVED CONTEXT:
{rag_context}

Explain:
1. Expected sensor behavior over next 24 hours
2. Any concerning trends
3. Optimal time window for maintenance
4. What could invalidate this forecast

Keep response under 150 words."""


# Combined template for a single per-run explanation (classification + RUL + anomaly + forecast)
COMBINED_RUN_PROMPT = """A predictive maintenance system ran multiple models for {machine_id}.

CURRENT SENSOR READINGS:
{sensor_readings}

BASELINE NORMAL RANGES (if available):
{baseline_ranges}

Interpretation:
- A sensor reading within its baseline min..max is normal.
- Only treat as overheating/overtemp if above baseline alarm or trip (or clearly above max).

MODEL OUTPUTS:
- Classification: label={failure_type} | label_probability={classification_probability} | failure_risk={failure_risk} | confidence={classification_confidence}
- RUL: rul_hours={rul_hours} | confidence={rul_confidence}
- Anomaly: anomaly_score={anomaly_score} | method={detection_method}
- Forecast: {forecast_summary}

RETRIEVED CONTEXT:
{rag_context}

Write ONE short maintenance brief in plain text (no Markdown).
Use exactly this structure and keep it under 160 words:

Overall: <one sentence>
Top causes:
- <cause 1 tied to specific sensor(s)>
- <cause 2 tied to specific sensor(s)>
Immediate actions:
- <action 1>
- <action 2>
- <action 3>
Next 7 days:
- <plan 1>
- <plan 2>
Safety: <one sentence>

Rules:
- Do NOT invent faults. Only call something a "cause" if it is supported by model outputs (high failure_risk, anomaly_score) OR current readings are outside the BASELINE NORMAL RANGES.
- Define "low" as: failure_risk < 0.15 AND anomaly_score < 0.30.
- For "normal" classification with low failure_risk and low anomaly_score, your Overall sentence MUST say the machine is operating normally with low risk.
- For "normal" classification with low failure_risk and low anomaly_score, your Top causes MUST be:
    - None detected (readings within baseline)
    - Continue monitoring
- Only describe overheating/overtemp when a temperature exceeds baseline max/alarm/trip.
- If you see a single temperature reading without variance/trend, do NOT call it "unstable".
- Avoid degree symbols; write temperatures as e.g. "210 C" (not "210°C").

If a model output is missing/unavailable, say "unavailable" and proceed."""


# Template usage example (for reference)
"""
Example Usage in Production (Phase 3.5):

from config.prompts import SYSTEM_PROMPT, FAILURE_CLASSIFICATION_PROMPT
from rag.retriever import MachineDocRetriever
from inference.test_llama import LlamaInference

# Get ML prediction from Phase 2 model
ml_prediction = {
    'machine_id': 'motor_siemens_1la7_001',
    'failure_type': 'bearing_wear',
    'probability': 0.87,
    'confidence': 0.92,
    'sensors': {
        'vibration': 12.5,
        'temperature': 78.0
    }
}

# Retrieve RAG context
retriever = MachineDocRetriever()
rag_results = retriever.retrieve(
    query=f"{ml_prediction['failure_type']} symptoms",
    machine_id=ml_prediction['machine_id'],
    top_k=3
)
rag_context = "\\n".join([r['doc'][:300] for r in rag_results])

# Format sensor readings
sensor_readings = f\"\"\"Vibration: {ml_prediction['sensors']['vibration']:.1f} mm/s (normal: <5)
Temperature: {ml_prediction['sensors']['temperature']:.1f}°C (normal: <65)\"\"\"

# Fill prompt template
user_message = FAILURE_CLASSIFICATION_PROMPT.format(
    machine_id=ml_prediction['machine_id'],
    failure_probability=ml_prediction['probability'],
    failure_type=ml_prediction['failure_type'],
    confidence=ml_prediction['confidence'],
    sensor_readings=sensor_readings,
    rag_context=rag_context
)

# Generate explanation
llm = LlamaInference()
explanation = llm.generate(SYSTEM_PROMPT, user_message)

# Result: Human-readable maintenance explanation
print(explanation)
"""
