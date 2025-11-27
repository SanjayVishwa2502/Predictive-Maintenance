"""
Test explanation quality with automated metrics
"""
import sys
import os
import json

# Add parent directory (LLM root) to path to allow importing api
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from api.explainer import MLExplainer

def test_explanation_quality():
    """Test multiple scenarios and evaluate"""
    
    print("Initializing MLExplainer...")
    explainer = MLExplainer()
    
    test_cases = [
        {
            'machine_id': 'motor_siemens_1la7_001',
            'failure_prob': 0.92,
            'failure_type': 'bearing_wear',
            'sensor_data': {'vibration': 15.2, 'temperature': 85.0},
            'expected_keywords': ['bearing', 'vibration', 'maintenance', 'replace']
        },
        {
            'machine_id': 'pump_grundfos_cr3_004',
            'failure_prob': 0.68,
            'failure_type': 'cavitation',
            'sensor_data': {'pressure': 2.1, 'flow_rate': 145.0},
            'expected_keywords': ['cavitation', 'pressure', 'inspect']
        }
    ]
    
    results = []
    
    print("Running test cases...")
    for case in test_cases:
        print(f"Testing {case['machine_id']}...")
        explanation = explainer.explain_classification(
            machine_id=case['machine_id'],
            failure_prob=case['failure_prob'],
            failure_type=case['failure_type'],
            sensor_data=case['sensor_data']
        )
        
        # Check quality metrics
        text = explanation['explanation'].lower()
        
        quality = {
            'machine_id': case['machine_id'],
            'length': len(text.split()),
            'keywords_found': sum(1 for kw in case['expected_keywords'] if kw in text),
            'keywords_total': len(case['expected_keywords']),
            'has_action': any(word in text for word in ['replace', 'inspect', 'check', 'monitor']),
            'has_safety': any(word in text for word in ['safety', 'shutdown', 'caution', 'risk']),
            'concise': len(text.split()) <= 250
        }
        
        quality['score'] = (
            (quality['keywords_found'] / quality['keywords_total']) * 0.4 +
            (1.0 if quality['has_action'] else 0.0) * 0.3 +
            (1.0 if quality['concise'] else 0.0) * 0.2 +
            (1.0 if quality['has_safety'] else 0.0) * 0.1
        )
        
        results.append(quality)
        
        print(f"\n{case['machine_id']}: Score {quality['score']:.2f}")
        print(f"  Words: {quality['length']}")
        print(f"  Keywords: {quality['keywords_found']}/{quality['keywords_total']}")
        print(f"  Action: {'✓' if quality['has_action'] else '✗'}")
        print(f"  Safety: {'✓' if quality['has_safety'] else '✗'}")
    
    # Save results
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports'))
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'explanation_quality_report.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    avg_score = sum(r['score'] for r in results) / len(results)
    print(f"\n=== AVERAGE QUALITY SCORE: {avg_score:.2f} ===")
    print(f"Report saved to {output_file}")

if __name__ == "__main__":
    test_explanation_quality()
