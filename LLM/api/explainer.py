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
    TIMESERIES_FORECAST_PROMPT
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
        print("\n" + "="*60)
        print("Initializing MLExplainer")
        print("="*60)
        
        print("\n[1/2] Loading LLM (Llama 3.1 8B)...")
        self.llm = LlamaInference()
        
        print("[2/2] Loading RAG Retriever...")
        self.retriever = MachineDocRetriever()
        
        print("="*60)
        print("✓ MLExplainer Ready")
        print("="*60)
        print()
    
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
        print(f"\n[Classification] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"{failure_type} symptoms in {machine_id}"
        print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        print(f"  ✓ Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Format sensor readings
        sensor_str = "\n".join([f"{k}: {v:.2f}" for k, v in sensor_data.items()])
        
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
        print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT, user_message, max_tokens=300
        )
        print(f"  ✓ Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
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
        print(f"\n[RUL] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"RUL estimation maintenance for {machine_id}"
        print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        print(f"  ✓ Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Format sensor readings
        sensor_str = "\n".join([f"{k}: {v:.2f}" for k, v in sensor_data.items()])
        
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
        print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT, user_message, max_tokens=300
        )
        print(f"  ✓ Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
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
        print(f"\n[Anomaly] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"anomaly investigation for {machine_id}"
        print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        print(f"  ✓ Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Format sensor readings
        sensor_str = "\n".join([f"{k}: {v:.2f}" for k, v in abnormal_sensors.items()])
        
        # Fill prompt
        user_message = ANOMALY_DETECTION_PROMPT.format(
            machine_id=machine_id,
            anomaly_score=anomaly_score,
            detection_method=detection_method,
            abnormal_sensors=sensor_str,
            rag_context=rag_context
        )
        
        # Generate explanation
        print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT, user_message, max_tokens=300
        )
        print(f"  ✓ Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
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
        print(f"\n[TimeSeries] Generating explanation for {machine_id}...")
        
        # Retrieve relevant context
        query = f"sensor forecasting for {machine_id}"
        print(f"  RAG Query: '{query}'")
        rag_results, elapsed_ms = self.retriever.retrieve(query, machine_id, top_k=2)
        rag_context = "\n\n".join([r['doc'][:400] for r in rag_results])
        print(f"  ✓ Retrieved {len(rag_results)} documents ({elapsed_ms:.0f}ms)")
        
        # Fill prompt
        user_message = TIMESERIES_FORECAST_PROMPT.format(
            machine_id=machine_id,
            forecast_summary=forecast_summary,
            confidence=confidence,
            rag_context=rag_context
        )
        
        # Generate explanation
        print("  Generating explanation with LLM...")
        explanation, inference_time, response_tokens, tokens_per_sec = self.llm.generate(
            SYSTEM_PROMPT, user_message, max_tokens=250
        )
        print(f"  ✓ Generated {len(explanation.split())} words ({inference_time:.1f}s, {tokens_per_sec:.0f} tok/s)")
        
        return {
            'explanation': explanation,
            'sources': [r['machine_id'] for r in rag_results],
            'confidence': confidence
        }


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
