"""
Prompt templates for ML explanations - Redesigned for Predictive Maintenance

Phase 3.5.1 - Revised: Data-driven, machine-aware prompt templates.
These templates use actual sensor data, baseline thresholds, and machine profiles
to generate contextually accurate, varied responses.
"""

# =============================================================================
# SYSTEM PROMPT - Machine-specific expert knowledge
# =============================================================================

SYSTEM_PROMPT = """You are an expert predictive maintenance AI assistant specializing in industrial equipment and 3D printers.

## Your Knowledge Base

### 3D Printer (Creality Ender 3) Operating Parameters:
- **Nozzle Temperature**: Normal 205-220 C during printing. Alarm at 230 C, Trip/Shutdown at 240 C.
- **Bed Temperature**: Normal 50-60 C during printing. Alarm at 65 C, Trip/Shutdown at 70 C.
- **Idle State**: Both temperatures near ambient (20-30 C) when not printing.
- **Common Issues**: Thermal runaway (temps exceed limits), heater failure (temps drop unexpectedly), thermistor drift.

### Industrial Motors (Siemens, ABB, WEG):
- **Vibration**: Normal <5 mm/s RMS. Warning 5-10 mm/s. Alarm >10 mm/s.
- **Temperature**: Normal <65 C. Warning 65-85 C. Alarm >85 C.
- **Current**: Depends on load. Sudden spikes indicate bearing issues or overload.

### CNC Machines (Haas, DMG Mori, Fanuc):
- **Spindle Temperature**: Normal 30-50 C. Warning >60 C.
- **Coolant Flow**: Normal >15 L/min. Low flow causes thermal issues.
- **Axis Load**: Normal <80%. Sustained >90% indicates tool wear.

## Critical Rules

1. **ALWAYS use the ACTUAL sensor values provided** - never invent readings.
2. **Compare readings against the BASELINE thresholds given** - this is your ground truth.
3. **Be specific**: Instead of "temperature is high", say "nozzle at 235 C exceeds 230 C alarm threshold".
4. **Status determines tone**:
   - If classification is "normal" with low failure_risk (<0.15): Report machine is healthy.
   - If anomaly_score < 0.30: No anomalies detected.
   - Only warn about issues when data supports it.
5. **Never contradict the data**: If sensors show normal values, do not describe problems.
6. **DETAILED RESPONSES**: Generate responses between 100-150 words with comprehensive analysis.
7. **No markdown formatting** - use plain text only.
8. **Use actual numbers** - "bed at 55 C (normal: 50-60 C)" not "bed temperature is within range".
9. **Include context**: Explain WHY readings are normal or abnormal, what they indicate about machine health, and what trends suggest.
"""


# =============================================================================
# FAILURE CLASSIFICATION PROMPT
# =============================================================================

FAILURE_CLASSIFICATION_PROMPT = """Analyze this machine's condition based on the ML prediction:

MACHINE: {machine_id}
PREDICTION: {failure_type} (probability: {failure_probability:.1%})
MODEL CONFIDENCE: {confidence:.1%}

CURRENT SENSOR READINGS:
{sensor_readings}

BASELINE NORMAL RANGES:
{baseline_ranges}

RETRIEVED CONTEXT:
{rag_context}

Generate a maintenance explanation. Compare EACH sensor reading against its baseline.
If all readings are within baseline and failure_probability < 15%, state the machine is operating normally.
If readings exceed thresholds, identify which specific sensors are abnormal and by how much.
Explain the operational implications and potential root causes.

Response should be 100-150 words. Plain text only, no markdown."""


# =============================================================================
# RUL REGRESSION PROMPT
# =============================================================================

RUL_REGRESSION_PROMPT = """Remaining Useful Life analysis for {machine_id}:

PREDICTED RUL: {rul_hours:.0f} hours ({rul_days:.1f} days)
CONFIDENCE: {confidence:.1%}

CURRENT SENSOR READINGS:
{sensor_readings}

BASELINE NORMAL RANGES:
{baseline_ranges}

RETRIEVED CONTEXT:
{rag_context}

Explain what this RUL prediction means for maintenance scheduling.
Reference SPECIFIC sensor values that influence the prediction.
If RUL > 500 hours, emphasize the machine is healthy with explanation of why.
If RUL < 100 hours, emphasize urgency with specific degradation indicators.
Include practical maintenance planning advice based on the RUL value.

Response should be 100-150 words. Plain text only."""


# =============================================================================
# ANOMALY DETECTION PROMPT
# =============================================================================

ANOMALY_DETECTION_PROMPT = """Anomaly analysis for {machine_id}:

ANOMALY SCORE: {anomaly_score:.2f} (threshold: 0.5)
DETECTION METHOD: {detection_method}

ABNORMAL SENSORS:
{abnormal_sensors}

BASELINE NORMAL RANGES:
{baseline_ranges}

RETRIEVED CONTEXT:
{rag_context}

Interpret the anomaly score:
- Score < 0.30: No significant anomaly detected. Machine operating normally. Explain what normal operation looks like.
- Score 0.30-0.50: Minor deviation. Monitor but no action needed. Explain what subtle patterns might be emerging.
- Score > 0.50: Significant anomaly. Investigate the specific sensors listed with root cause analysis.

Use the ACTUAL values from abnormal_sensors. Compare against baseline. Explain operational implications.

Response should be 100-150 words. Plain text only."""


# =============================================================================
# TIME-SERIES FORECAST PROMPT
# =============================================================================

TIMESERIES_FORECAST_PROMPT = """24-hour forecast for {machine_id}:

FORECAST SUMMARY:
{forecast_summary}

PREDICTION CONFIDENCE: {confidence:.1%}

CURRENT SENSOR READINGS:
{current_readings}

BASELINE NORMAL RANGES:
{baseline_ranges}

RETRIEVED CONTEXT:
{rag_context}

Explain the forecast in practical terms:
1. What sensor values are expected over the next 24 hours? Cite specific predicted values.
2. Will any values approach or exceed alarm thresholds? At what time?
3. Is this a stable forecast or trending toward a problem? Explain the trend direction.
4. What operational decisions should be made based on this forecast?

Use SPECIFIC predicted values from the forecast, not generic statements.

Response should be 100-150 words. Plain text only."""


# =============================================================================
# COMBINED RUN PROMPT - Primary template for dashboard explanations
# =============================================================================

COMBINED_RUN_PROMPT = """Analyze this predictive maintenance run for {machine_id}:

=== CURRENT SENSOR READINGS ===
{sensor_readings}

=== BASELINE NORMAL RANGES ===
{baseline_ranges}

=== ML MODEL OUTPUTS ===
Classification: {failure_type} | probability={classification_probability} | failure_risk={failure_risk} | confidence={classification_confidence}
RUL: {rul_hours} hours | confidence={rul_confidence}
Anomaly: score={anomaly_score} | method={detection_method}
Forecast: {forecast_summary}

=== CONTEXT ===
{rag_context}

=== ANALYSIS RULES ===
1. Compare EACH sensor value against its baseline min/max/alarm/trip thresholds.
2. "Normal" classification with failure_risk < 0.15 AND anomaly_score < 0.30 means machine is healthy.
3. Only flag sensors that ACTUALLY exceed their thresholds.
4. Use specific numbers: "nozzle at 212 C (typical: 210 C)" not "temperature is fine".
5. For 3D printers: nozzle around 210 C and bed around 55 C during printing is completely normal.

=== REQUIRED OUTPUT FORMAT (plain text, no markdown) ===

Overall: <comprehensive summary with actual readings, operational status, and overall health assessment>

Sensor Status:
- <sensor1>: <value> (<detailed status vs baseline, what this indicates>)
- <sensor2>: <value> (<detailed status vs baseline, what this indicates>)

{status_section}

Trend Analysis:
- <describe current trends and what they suggest about future machine health>

Next Steps:
- <action 1 with specific timeframe>
- <action 2 with reasoning>

Safety: <detailed safety assessment based on actual readings, or "No safety concerns - all parameters within operational limits.">

Generate a comprehensive response between 100-150 words that thoroughly analyzes the machine's condition."""


# Dynamic status section based on health
STATUS_SECTION_HEALTHY = """Risk Assessment:
- No faults detected (readings within baseline)
- Continue normal monitoring schedule"""

STATUS_SECTION_WARNING = """Top Concerns:
- <concern 1 with specific sensor and value>
- <concern 2 with specific sensor and value>"""


# =============================================================================
# PRINTER-SPECIFIC PROMPT (for Creality Ender 3)
# =============================================================================

PRINTER_COMBINED_PROMPT = """Analyze this 3D printer maintenance run:

PRINTER: {machine_id} (Creality Ender 3)
DATA TIMESTAMP: {data_stamp}

=== CURRENT TEMPERATURES ===
{sensor_readings}

=== NORMAL OPERATING RANGES (during printing) ===
Nozzle: min=205 C, typical=210 C, max=220 C, alarm=230 C, trip=240 C
Bed: min=50 C, typical=55 C, max=60 C, alarm=65 C, trip=70 C
(If idle/not printing, both temperatures will be near ambient 20-30 C)

=== ML PREDICTION ===
Status: {failure_type}
Failure Risk: {failure_risk}
Anomaly Score: {anomaly_score}
24h Forecast: {forecast_summary}

=== COMPREHENSIVE ANALYSIS (100-150 words required) ===

Output format requirements:
- Use Markdown (no HTML).
- Include one horizontal divider line using `---`.
- Include ONE compact Markdown table summarizing the temperatures.
- Keep the narrative explanation at 100-150 words (the table does not count toward the word target).

Provide a detailed maintenance brief covering:

1. Overall Status Assessment:
   - Is the printer operating normally? Use the ACTUAL temperature values.
    - If nozzle and bed are near ambient (20-30 C), explicitly state the printer appears IDLE and that this is expected.
    - Only compare against printing ranges when the temperatures indicate active heating.
    - Only flag as concerning if exceeding alarm thresholds (nozzle >230 C or bed >65 C)

2. Detailed Temperature Analysis:
   - Nozzle: <actual value> C - <normal/warning/alarm based on thresholds>
     Explain what this temperature indicates about extrusion quality and heater health.
   - Bed: <actual value> C - <normal/warning/alarm based on thresholds>
     Explain what this temperature indicates about adhesion and bed heater condition.

3. System Health Indicators:
   - Thermal stability assessment
   - Any thermal runaway risk indicators
   - PID tuning status based on temperature consistency

4. Operational Recommendations:
   - If all normal: "Continue printing operations. Thermal systems functioning optimally."
   - If warning: "Increase monitoring frequency. Check thermistor connections."
   - If alarm: "Pause print immediately. Inspect heating elements and wiring."

Markdown table requirement (place near the top, after a short one-line summary):

| Component | Current (°C) | Printing typical (°C) | Alarm (°C) | Interpretation |
|---|---:|---:|---:|---|
| Nozzle | <value> | 210 | 230 | <idle/heating/over-temp> |
| Bed | <value> | 55 | 65 | <idle/heating/over-temp> |

Generate a comprehensive response between 100-150 words (narrative only) using Markdown."""


# =============================================================================
# Helper function to select appropriate prompt based on machine type
# =============================================================================

def get_combined_prompt_for_machine(machine_id: str) -> str:
    """Return the most appropriate combined prompt template for a given machine."""
    machine_lower = (machine_id or "").lower()
    
    if "printer" in machine_lower or "ender" in machine_lower or "creality" in machine_lower:
        return PRINTER_COMBINED_PROMPT
    else:
        return COMBINED_RUN_PROMPT


def get_status_section(failure_risk: float, anomaly_score: float) -> str:
    """Return the appropriate status section based on risk levels."""
    try:
        fr = float(failure_risk) if failure_risk is not None else 0.0
        an = float(anomaly_score) if anomaly_score is not None else 0.0
    except (ValueError, TypeError):
        fr, an = 0.0, 0.0
    
    if fr < 0.15 and an < 0.30:
        return STATUS_SECTION_HEALTHY
    else:
        return STATUS_SECTION_WARNING
