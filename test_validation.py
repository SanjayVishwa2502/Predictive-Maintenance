"""
Test the MachineProfileValidator
This script tests validation with:
1. A valid new machine
2. Brother Speedio (duplicate - should FAIL)
3. A machine missing required fields (should FAIL)
"""

import json
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / 'frontend' / 'server'))

from api.services.profile_validator import MachineProfileValidator


def test_validation():
    """Test validation scenarios"""
    
    print("="*80)
    print("MACHINE PROFILE VALIDATION TEST")
    print("="*80)
    print()
    
    # Initialize validator
    validator = MachineProfileValidator()
    
    print(f"Existing machines in system: {len(validator.existing_machines)}")
    print(f"Sample existing machines: {list(validator.existing_machines)[:5]}")
    print()
    
    # Test 1: Brother Speedio (should FAIL - duplicate detection test)
    print("-" * 80)
    print("TEST 1: Brother Speedio Profile (Duplicate Check)")
    print("-" * 80)
    
    brother_profile_path = Path("cnc_brother_speedio_001.json")
    if brother_profile_path.exists():
        with open(brother_profile_path, 'r') as f:
            brother_profile = json.load(f)
        
        is_valid, issues, can_proceed = validator.validate_profile(brother_profile, strict=True)
        
        print(f"Machine ID: {brother_profile.get('machine_id')}")
        print(f"Validation Result: {'PASS' if is_valid else 'FAIL'}")
        print(f"Can Proceed: {can_proceed}")
        print(f"\nIssues Found: {len(issues)}")
        
        for issue in issues:
            icon = "🔴" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
            print(f"  {icon} [{issue.severity.upper()}] {issue.field}: {issue.message}")
        
        print()
        print(f"Expected: FAIL (duplicate machine_id)")
        print(f"Actual: {'FAIL ✓' if not is_valid else 'PASS ✗ (VALIDATION FAILED TO CATCH DUPLICATE!)'}")
    else:
        print(f"Brother Speedio profile not found at {brother_profile_path}")
    
    print()
    
    # Test 2: Valid new machine
    print("-" * 80)
    print("TEST 2: Valid New Machine Profile")
    print("-" * 80)
    
    valid_profile = {
        "machine_id": "test_new_machine_999",
        "manufacturer": "Test Manufacturer",
        "model": "Test Model 123",
        "category": "test_equipment",
        "sensors": [
            {"name": "temp_sensor_1", "min_value": 0, "max_value": 100},
            {"name": "vibration_sensor_1", "min_value": 0, "max_value": 50},
            {"name": "pressure_sensor_1", "min_value": 0, "max_value": 200},
        ],
        "rul_min": 0,
        "rul_max": 1000,
        "degradation_states": 4
    }
    
    is_valid, issues, can_proceed = validator.validate_profile(valid_profile, strict=True)
    
    print(f"Machine ID: {valid_profile.get('machine_id')}")
    print(f"Validation Result: {'PASS' if is_valid else 'FAIL'}")
    print(f"Can Proceed: {can_proceed}")
    print(f"\nIssues Found: {len(issues)}")
    
    for issue in issues:
        icon = "🔴" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
        print(f"  {icon} [{issue.severity.upper()}] {issue.field}: {issue.message}")
    
    print()
    print(f"Expected: PASS (new unique machine)")
    print(f"Actual: {'PASS ✓' if is_valid else f'FAIL ✗ (unexpected validation error)'}")
    
    print()
    
    # Test 3: Invalid machine (missing required fields)
    print("-" * 80)
    print("TEST 3: Invalid Machine Profile (Missing Required Fields)")
    print("-" * 80)
    
    invalid_profile = {
        "machine_id": "test_invalid_machine",
        # Missing: manufacturer, model, category, sensors
    }
    
    is_valid, issues, can_proceed = validator.validate_profile(invalid_profile, strict=True)
    
    print(f"Machine ID: {invalid_profile.get('machine_id')}")
    print(f"Validation Result: {'PASS' if is_valid else 'FAIL'}")
    print(f"Can Proceed: {can_proceed}")
    print(f"\nIssues Found: {len(issues)}")
    
    for issue in issues:
        icon = "🔴" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
        print(f"  {icon} [{issue.severity.upper()}] {issue.field}: {issue.message}")
    
    print()
    print(f"Expected: FAIL (missing required fields)")
    print(f"Actual: {'FAIL ✓' if not is_valid else 'PASS ✗ (validation should have failed!)'}")
    
    print()
    print("="*80)
    print("VALIDATION TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_validation()
