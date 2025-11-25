"""
Generate synthetic failure cases from ML model predictions

Phase 3.3.1: Create training/example data for LLM
Later (Phase 3.5): This will use REAL ML predictions from Phase 2 models

Current: Synthetic random cases for development
Future: Real predictions → LLM explanations
"""
import json
from pathlib import Path
import random


def generate_failure_case(machine_id, failure_type, rul, sensors):
    """Create synthetic failure case"""
    
    case = {
        'machine_id': machine_id,
        'failure_type': failure_type,
        'rul_hours': rul,
        'severity': 'high' if rul < 100 else 'medium' if rul < 300 else 'low',
        'sensor_readings': sensors,
        'symptoms': generate_symptoms(failure_type, sensors),
        'root_cause': generate_root_cause(failure_type),
        'corrective_action': generate_action(failure_type),
        'cost_impact': estimate_cost(failure_type, rul)
    }
    
    return case


def generate_symptoms(failure_type, sensors):
    """Generate realistic symptoms"""
    symptoms = {
        'bearing_wear': [
            f"Elevated vibration: {sensors.get('vibration', 0):.2f} mm/s",
            f"Increased temperature: {sensors.get('temperature', 0):.1f}°C",
            "Unusual noise detected"
        ],
        'overheating': [
            f"Temperature spike: {sensors.get('temperature', 0):.1f}°C",
            "Reduced cooling efficiency",
            "Thermal expansion detected"
        ],
        'electrical_fault': [
            f"Current imbalance: {sensors.get('current', 0):.2f}A",
            f"Voltage fluctuation: {sensors.get('voltage', 0):.1f}V",
            "Power factor degradation"
        ]
    }
    return symptoms.get(failure_type, ["Unknown symptoms"])


def generate_root_cause(failure_type):
    """Generate root cause analysis"""
    causes = {
        'bearing_wear': "Insufficient lubrication or contamination",
        'overheating': "Blocked cooling vents or ambient temperature too high",
        'electrical_fault': "Winding insulation degradation or loose connections"
    }
    return causes.get(failure_type, "Unknown cause")


def generate_action(failure_type):
    """Generate corrective action"""
    actions = {
        'bearing_wear': "Replace bearings, clean housing, relubricate",
        'overheating': "Clean cooling system, check ventilation, verify load",
        'electrical_fault': "Inspect wiring, test insulation resistance, tighten connections"
    }
    return actions.get(failure_type, "Inspect and diagnose")


def estimate_cost(failure_type, rul):
    """Estimate cost impact"""
    base_cost = {
        'bearing_wear': 2500,
        'overheating': 1500,
        'electrical_fault': 3500
    }
    
    cost = base_cost.get(failure_type, 2000)
    
    # Emergency cost if RUL < 24 hours
    if rul < 24:
        cost *= 2.5
    
    return f"${cost:,.0f}"


def batch_generate_cases():
    """Generate 100+ synthetic cases"""
    
    failure_types = ['bearing_wear', 'overheating', 'electrical_fault']
    machines = [
        'motor_siemens_1la7_001', 'motor_abb_m3bp_002', 
        'pump_grundfos_cr3_004', 'compressor_atlas_copco_ga30_001'
    ]
    
    cases = []
    for i in range(100):
        machine = random.choice(machines)
        failure = random.choice(failure_types)
        rul = random.uniform(10, 500)
        sensors = {
            'vibration': random.uniform(0.5, 15.0),
            'temperature': random.uniform(40, 95),
            'current': random.uniform(10, 50),
            'voltage': random.uniform(380, 420)
        }
        
        case = generate_failure_case(machine, failure, rul, sensors)
        cases.append(case)
    
    # Save
    output_file = Path("../../data/knowledge_base/failure_cases.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(cases, f, indent=2)
    
    print(f"✓ Generated {len(cases)} failure cases")
    print(f"✓ Saved to {output_file}")
    print(f"\n📝 NOTE: These are SYNTHETIC cases for development")
    print(f"   In Phase 3.5, we'll use REAL ML predictions from Phase 2 models")
    print(f"   Final workflow: Sensors → ML Models → LLM Explanations → User")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3.3.1: Generate Synthetic Failure Cases")
    print("=" * 60)
    print("\nPurpose: Create training examples for LLM")
    print("Stage: Development (synthetic data)")
    print("Later: Production (real ML predictions)\n")
    
    batch_generate_cases()
    
    print("\n" + "=" * 60)
    print("✅ Phase 3.3.1 Complete!")
    print("=" * 60)
