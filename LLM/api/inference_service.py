"""
Unified Inference Service - ML → LLM Pipeline
Phase 3.5.1: MLExplainer API Implementation

Orchestrates the complete pipeline:
1. Load ML models (classification, anomaly, RUL, timeseries)
2. Run ML inference on sensor data
3. Retrieve relevant context from RAG knowledge base
4. Format prompt with ML predictions + RAG context
5. Generate human-readable explanation with LLM

This is a skeleton implementation - full implementation in Phase 3.5.1
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


class MLModelManager:
    """Manages lazy loading and caching of ML models"""
    
    def __init__(self, models_dir: Path, max_cached: int = 5):
        """
        Initialize model manager
        
        Args:
            models_dir: Root directory containing all ML models
            max_cached: Maximum number of models to keep in memory (LRU)
        """
        self.models_dir = models_dir
        self.max_cached = max_cached
        self.cache = {}  # {(machine_id, model_type): model_instance}
        self.access_times = {}  # Track last access for LRU
        
    def load_model(self, machine_id: str, model_type: str):
        """
        Load ML model with caching (lazy load)
        
        Args:
            machine_id: Machine identifier
            model_type: 'classification', 'anomaly', 'rul', or 'timeseries'
            
        Returns:
            Model instance
        """
        cache_key = (machine_id, model_type)
        
        # Check cache
        if cache_key in self.cache:
            self.access_times[cache_key] = datetime.utcnow()
            return self.cache[cache_key]
        
        # Load model based on type
        if model_type == 'classification':
            from ml_models.scripts.inference.predict_classification import ClassificationPredictor
            model = ClassificationPredictor(machine_id)
        elif model_type == 'anomaly':
            from ml_models.scripts.inference.predict_anomaly import AnomalyPredictor
            model = AnomalyPredictor(machine_id)
        elif model_type == 'rul':
            from ml_models.scripts.inference.predict_rul import RULPredictor
            model = RULPredictor(machine_id)
        elif model_type == 'timeseries':
            from ml_models.scripts.inference.predict_timeseries import TimeSeriesPredictor
            model = TimeSeriesPredictor(machine_id)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Add to cache
        self.cache[cache_key] = model
        self.access_times[cache_key] = datetime.utcnow()
        
        # Evict oldest if cache full
        if len(self.cache) > self.max_cached:
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        return model
    
    def clear_cache(self):
        """Clear all cached models"""
        self.cache.clear()
        self.access_times.clear()


class RAGRetriever:
    """Retrieves relevant context from knowledge base using RAG"""
    
    def __init__(self, embeddings_dir: Path):
        """
        Initialize RAG retriever
        
        Args:
            embeddings_dir: Directory containing FAISS index and documents
        """
        self.embeddings_dir = embeddings_dir
        self.index = None
        self.documents = []
        self.embedder = None
        
    def load_index(self):
        """Load FAISS index and documents"""
        # TODO: Implement in Phase 3.5.1
        # from sentence_transformers import SentenceTransformer
        # import faiss
        # self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # self.index = faiss.read_index(str(self.embeddings_dir / 'knowledge_base.index'))
        # with open(self.embeddings_dir / 'documents.json') as f:
        #     self.documents = json.load(f)
        pass
    
    def retrieve(self, query: str, machine_id: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents
        
        Args:
            query: Search query (e.g., "bearing wear high vibration")
            machine_id: Machine identifier for filtering
            top_k: Number of documents to retrieve
            
        Returns:
            List of relevant documents with metadata
        """
        # TODO: Implement in Phase 3.5.1
        # query_embedding = self.embedder.encode([query])
        # distances, indices = self.index.search(query_embedding, top_k)
        # results = [self.documents[i] for i in indices[0]]
        # return results
        return []


class PromptFormatter:
    """Formats prompts using templates"""
    
    def __init__(self, templates_dir: Path):
        """
        Initialize prompt formatter
        
        Args:
            templates_dir: Directory containing prompt templates
        """
        self.templates_dir = templates_dir
        self.templates = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all prompt templates"""
        # TODO: Implement in Phase 3.5.1
        # Load templates from templates_dir
        pass
    
    def format_prompt(self, model_type: str, ml_prediction: Dict, 
                     rag_context: List[Dict], machine_id: str) -> str:
        """
        Format prompt by filling template
        
        Args:
            model_type: Type of ML model
            ml_prediction: ML model prediction output
            rag_context: Retrieved context documents
            machine_id: Machine identifier
            
        Returns:
            Formatted prompt string
        """
        # TODO: Implement in Phase 3.5.1
        # template = self.templates[model_type]
        # return template.format(
        #     machine_id=machine_id,
        #     prediction=ml_prediction,
        #     context=rag_context
        # )
        return ""


class LLMGenerator:
    """Generates explanations using LLM"""
    
    def __init__(self, model_path: Path):
        """
        Initialize LLM generator
        
        Args:
            model_path: Path to GGUF model file
        """
        self.model_path = model_path
        self.llm = None
        
    def load_model(self):
        """Load LLM model"""
        # TODO: Implement in Phase 3.5.1
        # from llama_cpp import Llama
        # self.llm = Llama(
        #     model_path=str(self.model_path),
        #     n_ctx=4096,
        #     n_threads=4,
        #     n_gpu_layers=0
        # )
        pass
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """
        Generate explanation from prompt
        
        Args:
            prompt: Formatted prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated explanation text
        """
        # TODO: Implement in Phase 3.5.1
        # response = self.llm(
        #     prompt,
        #     max_tokens=max_tokens,
        #     temperature=0.7,
        #     top_p=0.9
        # )
        # return response['choices'][0]['text']
        return ""


class UnifiedInferenceService:
    """
    Main service orchestrating ML → RAG → LLM pipeline
    
    This is the central component that brings everything together.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize unified inference service
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = config or self._load_default_config()
        
        # Initialize components
        self.ml_manager = MLModelManager(
            models_dir=PROJECT_ROOT / "ml_models" / "models",
            max_cached=self.config.get('max_cached_models', 5)
        )
        
        self.rag_retriever = RAGRetriever(
            embeddings_dir=PROJECT_ROOT / "LLM" / "data" / "embeddings"
        )
        
        self.prompt_formatter = PromptFormatter(
            templates_dir=PROJECT_ROOT / "LLM" / "templates"
        )
        
        self.llm_generator = LLMGenerator(
            model_path=PROJECT_ROOT / "LLM" / "models" / self.config.get('llm_model_file')
        )
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'avg_latency_ms': 0
        }
    
    def _load_default_config(self) -> Dict:
        """Load default configuration"""
        return {
            'max_cached_models': 5,
            'llm_model_file': 'llama-3.1-8b-instruct-q4_k_m.gguf',
            'rag_top_k': 5,
            'llm_max_tokens': 512,
            'timeout_seconds': 65
        }
    
    def initialize(self):
        """
        Initialize all components (load models, indices, etc.)
        Call this at startup before handling requests
        """
        print("Initializing Unified Inference Service...")
        
        # Load RAG index
        print("  Loading RAG index...")
        self.rag_retriever.load_index()
        
        # Load LLM model
        print("  Loading LLM model...")
        self.llm_generator.load_model()
        
        # Preload most-used ML models (optional)
        # self._preload_common_models()
        
        print("✓ Unified Inference Service initialized")
    
    def run_ml_inference(self, machine_id: str, sensor_data: Dict, 
                        model_type: str) -> Dict:
        """
        Run ML model inference
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dictionary of sensor readings
            model_type: Type of model to use
            
        Returns:
            ML prediction output
        """
        model = self.ml_manager.load_model(machine_id, model_type)
        
        if model_type == 'timeseries':
            # Time-series requires historical data
            from ml_models.scripts.inference.predict_timeseries import create_sample_historical_data
            historical_data = create_sample_historical_data(machine_id)
            prediction = model.predict(historical_data)
        else:
            # Classification, anomaly, RUL use current sensor data
            prediction = model.predict(sensor_data)
        
        return prediction
    
    def retrieve_context(self, query: str, machine_id: str) -> List[Dict]:
        """
        Retrieve relevant context from knowledge base
        
        Args:
            query: Search query
            machine_id: Machine identifier
            
        Returns:
            List of relevant documents
        """
        return self.rag_retriever.retrieve(
            query=query,
            machine_id=machine_id,
            top_k=self.config['rag_top_k']
        )
    
    def format_prompt(self, model_type: str, ml_prediction: Dict,
                     rag_context: List[Dict], machine_id: str) -> str:
        """
        Format prompt for LLM
        
        Args:
            model_type: Type of ML model
            ml_prediction: ML prediction output
            rag_context: Retrieved context
            machine_id: Machine identifier
            
        Returns:
            Formatted prompt
        """
        return self.prompt_formatter.format_prompt(
            model_type=model_type,
            ml_prediction=ml_prediction,
            rag_context=rag_context,
            machine_id=machine_id
        )
    
    def generate_explanation(self, prompt: str) -> str:
        """
        Generate explanation using LLM
        
        Args:
            prompt: Formatted prompt
            
        Returns:
            Generated explanation
        """
        return self.llm_generator.generate(
            prompt=prompt,
            max_tokens=self.config['llm_max_tokens']
        )
    
    def explain(self, machine_id: str, sensor_data: Dict, 
               model_type: str) -> Dict:
        """
        End-to-end pipeline: ML → RAG → LLM → Explanation
        
        Args:
            machine_id: Machine identifier
            sensor_data: Dictionary of sensor readings
            model_type: Type of model ('classification', 'anomaly', 'rul', 'timeseries')
            
        Returns:
            Complete explanation with metadata
        """
        start_time = datetime.utcnow()
        request_id = f"{machine_id}_{int(start_time.timestamp())}"
        
        try:
            # Step 1: ML Inference
            ml_start = datetime.utcnow()
            ml_prediction = self.run_ml_inference(machine_id, sensor_data, model_type)
            ml_time = (datetime.utcnow() - ml_start).total_seconds() * 1000
            
            # Step 2: RAG Retrieval
            rag_start = datetime.utcnow()
            query = self._build_rag_query(ml_prediction, model_type)
            rag_context = self.retrieve_context(query, machine_id)
            rag_time = (datetime.utcnow() - rag_start).total_seconds() * 1000
            
            # Step 3: Format Prompt
            prompt = self.format_prompt(model_type, ml_prediction, rag_context, machine_id)
            
            # Step 4: LLM Generation
            llm_start = datetime.utcnow()
            explanation = self.generate_explanation(prompt)
            llm_time = (datetime.utcnow() - llm_start).total_seconds() * 1000
            
            # Calculate total latency
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Update stats
            self.stats['total_requests'] += 1
            self.stats['successful'] += 1
            self.stats['avg_latency_ms'] = (
                (self.stats['avg_latency_ms'] * (self.stats['successful'] - 1) + total_time)
                / self.stats['successful']
            )
            
            # Construct response
            result = {
                'request_id': request_id,
                'timestamp': start_time.isoformat() + 'Z',
                'machine_id': machine_id,
                'model_type': model_type,
                'ml_prediction': ml_prediction.get('prediction', {}),
                'explanation': {
                    'text': explanation,
                    # TODO: Parse explanation to extract key_findings, root_cause, recommendations
                },
                'metadata': {
                    'ml_inference_time_ms': round(ml_time, 2),
                    'rag_retrieval_time_ms': round(rag_time, 2),
                    'llm_generation_time_ms': round(llm_time, 2),
                    'total_latency_ms': round(total_time, 2),
                    'context_documents': len(rag_context),
                    'prompt_template': f"{model_type}_v1"
                }
            }
            
            return result
            
        except Exception as e:
            # Error handling
            self.stats['total_requests'] += 1
            self.stats['failed'] += 1
            
            return {
                'request_id': request_id,
                'timestamp': start_time.isoformat() + 'Z',
                'machine_id': machine_id,
                'model_type': model_type,
                'error': str(e),
                'status': 'failed'
            }
    
    def _build_rag_query(self, ml_prediction: Dict, model_type: str) -> str:
        """
        Build RAG query from ML prediction
        
        Args:
            ml_prediction: ML prediction output
            model_type: Type of model
            
        Returns:
            Query string for RAG retrieval
        """
        # TODO: Implement smarter query building in Phase 3.5.1
        if model_type == 'classification':
            failure_type = ml_prediction.get('prediction', {}).get('failure_type', 'unknown')
            return f"{failure_type} failure diagnosis"
        elif model_type == 'anomaly':
            severity = ml_prediction.get('prediction', {}).get('severity', 'unknown')
            return f"{severity} severity anomaly detection"
        elif model_type == 'rul':
            urgency = ml_prediction.get('prediction', {}).get('urgency', 'unknown')
            return f"{urgency} urgency remaining useful life prediction"
        else:
            return "time series forecasting maintenance planning"
    
    def get_stats(self) -> Dict:
        """Get service statistics"""
        return self.stats


# Singleton instance
_service_instance = None


def get_service(config: Optional[Dict] = None) -> UnifiedInferenceService:
    """
    Get or create unified inference service singleton
    
    Args:
        config: Configuration dictionary (optional)
        
    Returns:
        UnifiedInferenceService instance
    """
    global _service_instance
    
    if _service_instance is None:
        _service_instance = UnifiedInferenceService(config)
        _service_instance.initialize()
    
    return _service_instance


def main():
    """Main function for testing"""
    print("="*60)
    print("UNIFIED INFERENCE SERVICE - SKELETON")
    print("="*60)
    print("\nThis is a skeleton implementation.")
    print("Full implementation will be completed in Phase 3.5.1")
    print("\nComponents:")
    print("  ✓ MLModelManager - ML model loading and caching")
    print("  ✓ RAGRetriever - Knowledge base retrieval")
    print("  ✓ PromptFormatter - Prompt template formatting")
    print("  ✓ LLMGenerator - Explanation generation")
    print("  ✓ UnifiedInferenceService - End-to-end orchestration")
    print("\nNext Steps (Phase 3.5.1):")
    print("  1. Implement RAGRetriever.retrieve()")
    print("  2. Implement PromptFormatter.format_prompt()")
    print("  3. Implement LLMGenerator.generate()")
    print("  4. Test end-to-end pipeline")
    print("  5. Create FastAPI endpoints")
    print("="*60)


if __name__ == "__main__":
    main()
