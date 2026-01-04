"""
Unified LLM explainer for all ML model types
Phase 3.5.1: API Design (Days 1-2)

This module provides the MLExplainer class that generates human-readable
explanations for ML predictions using:
- RAG retrieval for relevant machine documentation
- Llama 3.1 8B for natural language generation
- Structured prompt templates for each model type
"""
from pathlib import Path
import sys
import os

# Add parent directories to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root / "LLM" / "scripts" / "inference"))
sys.path.append(str(project_root / "LLM" / "scripts" / "rag"))
sys.path.append(str(project_root / "LLM" / "config"))

from llama_engine import LlamaInference
from retriever import MachineDocRetriever
from prompts import (
    SYSTEM_PROMPT,
    FAILURE_CLASSIFICATION_PROMPT,
    RUL_REGRESSION_PROMPT,
    ANOMALY_DETECTION_PROMPT,
    TIMESERIES_FORECAST_PROMPT,
    COMBINED_RUN_PROMPT,
    PRINTER_COMBINED_PROMPT,
    get_combined_prompt_for_machine,
    get_status_section,
)


class MLExplainer:
    """
    Unified explainer for all ML model types.
    
    Generates human-readable maintenance explanations by combining:
    - ML model predictions
    - RAG-retrieved context
    - LLM natural language generation
    """
    
    def __init__(self):
        """Initialize LLM and RAG retriever"""
        self.verbose = (os.getenv("PM_LLM_VERBOSE", "0").strip() == "1")

        if self.verbose:
            print("\n" + "="*60)
            print("Initializing MLExplainer")
            print("="*60)
            print("\n[1/2] Loading LLM (Llama 3.1 8B)...")
        self.llm = LlamaInference()

        if self.verbose:
            print("[2/2] Loading RAG Retriever...")
        self.retriever = MachineDocRetriever()

        if self.verbose:
            print("="*60)
            print("[OK] MLExplainer Ready")
            print("="*60)
            print()

    def _max_tokens(self, env_name: str, default: int) -> int:
        try:
            v = int(os.getenv(env_name, str(default)).strip())
            return max(64, min(v, 800))  # Increased cap for detailed responses
        except Exception:
            return default

    def _sensor_str(self, sensor_data) -> str:
        try:
            items = list((sensor_data or {}).items())
        except Exception:
            items = []
        # Keep prompt small: cap to first 25 sensors
        lines = []
        for k, v in items[:25]:
            try:
                lines.append(f"{k}: {float(v):.2f}")
            except Exception:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)
    
    def explain_classification(self, machine_id, failure_prob, failure_type, 
                               sensor_data, confidence=0.9):
        """
        Explain failure classification prediction
        
        Args:
            machine_id: Machine identifier (e.g., "motor_siemens_1la7_001")
            failure_prob: Predicted failure probability (0.0-1.0)
            failure_type: Predicted failure type (e.g., "bearing_wear")
            sensor_data: Dict of current sensor readings
            confidence: Model confidence score (0.0-1.0)
        
        Returns:
            Dict with:
                - explanation: Human-readable text
                - sources: List of machine_ids used for context
                - confidence: Model confidence score
        """
        if self.verbose:
            print(f"\n[Classification] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"{failure_type} symptoms in {machine_id}"
        if self.verbose:
            print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        if self.verbose:
            print(f"  [OK] Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Format sensor readings
        sensor_str = self._sensor_str(sensor_data)
        
        # Fill prompt
        user_message = FAILURE_CLASSIFICATION_PROMPT.format(
            machine_id=machine_id,
            failure_probability=failure_prob,
            failure_type=failure_type,
            confidence=confidence,
            sensor_readings=sensor_str,
            rag_context=rag_context
        )
        
        # Generate explanation
        if self.verbose:
            print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT,
            user_message,
            max_tokens=self._max_tokens("PM_LLM_MAX_TOKENS_CLASSIFICATION", 350),  # Increased for detailed responses
        )
        if self.verbose:
            print(f"  [OK] Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'confidence': confidence
        }
    
    def explain_rul(self, machine_id, rul_hours, sensor_data, confidence=0.9):
        """
        Explain RUL (Remaining Useful Life) prediction
        
        Args:
            machine_id: Machine identifier
            rul_hours: Predicted remaining useful life in hours
            sensor_data: Dict of current sensor readings
            confidence: Model confidence score (0.0-1.0)
        
        Returns:
            Dict with:
                - explanation: Human-readable text
                - sources: List of machine_ids used for context
                - confidence: Model confidence score
        """
        if self.verbose:
            print(f"\n[RUL] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"RUL estimation maintenance for {machine_id}"
        if self.verbose:
            print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        if self.verbose:
            print(f"  [OK] Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Format sensor readings
        sensor_str = self._sensor_str(sensor_data)
        
        # Fill prompt
        user_message = RUL_REGRESSION_PROMPT.format(
            machine_id=machine_id,
            rul_hours=rul_hours,
            rul_days=rul_hours / 24,
            confidence=confidence,
            sensor_readings=sensor_str,
            rag_context=rag_context
        )
        
        # Generate explanation
        if self.verbose:
            print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT,
            user_message,
            max_tokens=self._max_tokens("PM_LLM_MAX_TOKENS_RUL", 350),
        )
        if self.verbose:
            print(f"  [OK] Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'confidence': confidence
        }
    
    def explain_anomaly(self, machine_id, anomaly_score, abnormal_sensors, 
                        detection_method, threshold=0.5):
        """
        Explain anomaly detection result
        
        Args:
            machine_id: Machine identifier
            anomaly_score: Anomaly score (0.0-1.0, higher = more anomalous)
            abnormal_sensors: Dict of sensors with abnormal readings
            detection_method: Detection algorithm used (e.g., "Isolation Forest")
            threshold: Anomaly threshold (0.0-1.0)
        
        Returns:
            Dict with:
                - explanation: Human-readable text
                - sources: List of machine_ids used for context
                - anomaly_score: Score from model
                - threshold: Detection threshold
        """
        if self.verbose:
            print(f"\n[Anomaly] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"anomaly investigation for {machine_id}"
        if self.verbose:
            print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        if self.verbose:
            print(f"  [OK] Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Format sensor readings
        sensor_str = self._sensor_str(abnormal_sensors)
        
        # Fill prompt
        user_message = ANOMALY_DETECTION_PROMPT.format(
            machine_id=machine_id,
            anomaly_score=anomaly_score,
            detection_method=detection_method,
            abnormal_sensors=sensor_str,
            rag_context=rag_context
        )
        
        # Generate explanation
        if self.verbose:
            print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT,
            user_message,
            max_tokens=self._max_tokens("PM_LLM_MAX_TOKENS_ANOMALY", 350),
        )
        if self.verbose:
            print(f"  [OK] Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'anomaly_score': anomaly_score,
            'threshold': threshold
        }
    
    def explain_forecast(self, machine_id, forecast_summary, confidence=0.85):
        """
        Explain time-series forecast
        
        Args:
            machine_id: Machine identifier
            forecast_summary: Summary of forecast (e.g., "Temperature rising 5°C over 24h")
            confidence: Model confidence score (0.0-1.0)
        
        Returns:
            Dict with:
                - explanation: Human-readable text
                - sources: List of machine_ids used for context
                - confidence: Model confidence score
        """
        if self.verbose:
            print(f"\n[TimeSeries] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"sensor forecasting for {machine_id}"
        if self.verbose:
            print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        if self.verbose:
            print(f"  [OK] Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Fill prompt
        user_message = TIMESERIES_FORECAST_PROMPT.format(
            machine_id=machine_id,
            forecast_summary=forecast_summary,
            confidence=confidence,
            rag_context=rag_context
        )
        
        # Generate explanation
        if self.verbose:
            print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT,
            user_message,
            max_tokens=self._max_tokens("PM_LLM_MAX_TOKENS_TIMESERIES", 350),
        )
        if self.verbose:
            print(f"  [OK] Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'confidence': confidence
        }

    def explain_combined_run(
        self,
        machine_id: str,
        predictions: dict,
        sensor_data: dict,
        baseline_ranges: str | None = None,
        data_stamp: str | None = None,
    ) -> dict:
        """Generate ONE combined explanation for a run.

        This is the main lever to reduce end-to-end latency on CPU: one LLM call per
        run rather than 3-4 separate ones.
        
        REDESIGNED: Now uses machine-specific prompts and properly formats sensor data
        with actual values and thresholds for accurate, varied responses.
        """
        if self.verbose:
            print(f"\n[Combined] Generating explanation for {machine_id}...")
            print(f"  Sensor data keys: {list((sensor_data or {}).keys())}")

        cls = (predictions or {}).get("classification") or {}
        rul = (predictions or {}).get("rul") or {}
        ano = (predictions or {}).get("anomaly") or {}
        ts = (predictions or {}).get("timeseries") or {}

        # If a predictor failed, avoid feeding error blobs into the prompt.
        if isinstance(cls, dict) and cls.get("error"):
            cls = {}
        if isinstance(rul, dict) and rul.get("error"):
            rul = {}
        if isinstance(ano, dict) and ano.get("error"):
            ano = {}
        if isinstance(ts, dict) and ts.get("error"):
            ts = {}

        failure_type = cls.get("failure_type") or cls.get("predicted_failure_type") or "normal"
        cls_conf = cls.get("confidence")
        classification_probability = cls_conf
        failure_risk = cls.get("failure_probability") or 0.0

        rul_hours = rul.get("rul_hours")
        rul_conf = rul.get("confidence")

        anomaly_score = ano.get("anomaly_score") or 0.0
        detection_method = ano.get("detection_method") or "Isolation Forest"

        forecast_summary = ts.get("forecast_summary") or ts.get("summary") or "(no forecast available)"

        # Retrieve relevant context once
        query = f"maintenance summary for {machine_id}"
        rag_results, _elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results]) or "No additional context available."

        # Format sensor data with actual values
        sensor_str = self._format_sensor_data_detailed(sensor_data)

        # Determine status section based on risk levels
        status_section = get_status_section(failure_risk, anomaly_score)

        # Select the appropriate prompt template based on machine type
        machine_lower = (machine_id or "").lower()
        is_printer = "printer" in machine_lower or "ender" in machine_lower or "creality" in machine_lower

        if is_printer:
            # Use printer-specific prompt with actual temperature values
            user_message = PRINTER_COMBINED_PROMPT.format(
                machine_id=machine_id,
                data_stamp=data_stamp or "current",
                sensor_readings=sensor_str,
                failure_type=failure_type,
                failure_risk=f"{float(failure_risk):.2f}" if failure_risk else "0.00",
                anomaly_score=f"{float(anomaly_score):.2f}" if anomaly_score else "0.00",
                forecast_summary=forecast_summary,
            )
        else:
            # Use general combined prompt
            user_message = COMBINED_RUN_PROMPT.format(
                machine_id=machine_id,
                sensor_readings=sensor_str,
                baseline_ranges=(baseline_ranges or "unavailable"),
                failure_type=failure_type,
                classification_probability=classification_probability,
                failure_risk=failure_risk,
                classification_confidence=cls_conf,
                rul_hours=rul_hours,
                rul_confidence=rul_conf,
                anomaly_score=anomaly_score,
                detection_method=detection_method,
                forecast_summary=forecast_summary,
                rag_context=rag_context,
                status_section=status_section,
            )

        if self.verbose:
            print(f"  Using {'PRINTER' if is_printer else 'GENERAL'} prompt template")
            print(f"  Prompt length: {len(user_message)} chars")

        explanation, inference_time, _response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT,
            user_message,
            max_tokens=self._max_tokens("PM_LLM_MAX_TOKENS_COMBINED", 450),  # Increased for 100-150 word responses
        )
        if self.verbose:
            print(f"  [OK] Combined explanation ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")

        return {
            "explanation": explanation,
            "sources": [r['machine_id'] for r in rag_results],
            "confidence": cls_conf or rul_conf or 0.0,
        }

    def _format_sensor_data_detailed(self, sensor_data: dict) -> str:
        """Format sensor data with actual values for the LLM prompt.
        
        This ensures the LLM sees specific numbers it must use in its response.
        """
        if not sensor_data:
            return "No sensor data available"
        
        lines = []
        for key, value in list(sensor_data.items())[:20]:  # Limit to 20 sensors
            # Skip non-numeric or internal fields
            if key in ('timestamp', 'machine_id', 'label', 'health_state'):
                continue
            
            try:
                val = float(value)
                # Format based on sensor type
                if 'temp' in str(key).lower():
                    lines.append(f"{key}: {val:.1f} C")
                elif 'pressure' in str(key).lower():
                    lines.append(f"{key}: {val:.2f} bar")
                elif 'vibration' in str(key).lower():
                    lines.append(f"{key}: {val:.2f} mm/s")
                elif 'current' in str(key).lower():
                    lines.append(f"{key}: {val:.2f} A")
                elif 'voltage' in str(key).lower():
                    lines.append(f"{key}: {val:.1f} V")
                elif 'speed' in str(key).lower() or 'rpm' in str(key).lower():
                    lines.append(f"{key}: {val:.0f} RPM")
                elif 'flow' in str(key).lower():
                    lines.append(f"{key}: {val:.2f} L/min")
                else:
                    lines.append(f"{key}: {val:.2f}")
            except (ValueError, TypeError):
                # Keep string values as-is
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines) if lines else "No numeric sensor data"


# Test the MLExplainer
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING MLExplainer - Phase 3.5.1")
    print("="*60)
    
    # Initialize explainer
    explainer = MLExplainer()
    
    # Test classification explanation
    print("\n" + "="*60)
    print("TEST 1: Classification Explanation")
    print("="*60)
    
    result = explainer.explain_classification(
        machine_id="motor_siemens_1la7_001",
        failure_prob=0.87,
        failure_type="bearing_wear",
        sensor_data={'vibration': 12.5, 'temperature': 78.0, 'current': 45.2},
        confidence=0.92
    )
    
    print("\n" + "="*60)
    print("CLASSIFICATION EXPLANATION")
    print("="*60)
    print(result['explanation'])
    print("\n" + "-"*60)
    print(f"Sources: {result['sources']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("="*60)
