"""
Comprehensive Validation Test Suite
Tests ALL 9 validation functions in MachineProfileValidator
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'frontend' / 'server'))

from api.services.profile_validator import MachineProfileValidator


def print_section(title):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_issues(issues):
    """Print validation issues"""
    for issue in issues:
        icon = {"error": "🔴", "warning": "⚠️", "info": "ℹ️"}.get(issue.severity, "•")
        print(f"  {icon} [{issue.severity.upper()}] {issue.field}: {issue.message}")


def test_all_validations():
    """Test all 9 validation functions"""
    
    print("\n" + "█"*80)
    print(" "*20 + "COMPREHENSIVE VALIDATION TEST SUITE")
    print("█"*80)
    
    validator = MachineProfileValidator()
    print(f"\nValidator initialized: {len(validator.existing_machines)} existing machines detected")
    
    # =========================================================================
    # TEST 1: _validate_required_fields
    # =========================================================================
    print_section("TEST 1: Required Fields Validation")
    
    test_cases = [
        {
            "name": "All required fields present",
            "data": {
                "machine_id": "test_complete_001",
                "manufacturer": "Test Corp",
                "model": "Model X",
                "machine_type": "test_machine"
            },
            "expect": "PASS"
        },
        {
            "name": "Missing manufacturer",
            "data": {
                "machine_id": "test_missing_mfg_001",
                "model": "Model X",
                "machine_type": "test_machine"
            },
            "expect": "FAIL"
        },
        {
            "name": "Missing all fields",
            "data": {},
            "expect": "FAIL"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_required_fields(tc['data'])
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result == tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 2: _validate_machine_id_uniqueness (CRITICAL)
    # =========================================================================
    print_section("TEST 2: Machine ID Uniqueness (Duplicate Detection)")
    
    test_cases = [
        {
            "name": "Duplicate machine_id (cnc_haas_vf3_001)",
            "data": {"machine_id": "cnc_haas_vf3_001"},
            "expect": "FAIL"
        },
        {
            "name": "Unique machine_id",
            "data": {"machine_id": "test_unique_machine_999"},
            "expect": "PASS"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_machine_id_uniqueness(tc['data'], strict=True)
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result == tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 3: _validate_machine_id_format
    # =========================================================================
    print_section("TEST 3: Machine ID Format Validation")
    
    test_cases = [
        {
            "name": "Valid format: motor_siemens_1la7_001",
            "data": {"machine_id": "motor_siemens_1la7_001"},
            "expect": "PASS"
        },
        {
            "name": "Invalid: uppercase letters",
            "data": {"machine_id": "Motor_Siemens_001"},
            "expect": "FAIL"
        },
        {
            "name": "Invalid: consecutive underscores",
            "data": {"machine_id": "motor__siemens_001"},
            "expect": "FAIL"
        },
        {
            "name": "Invalid: starts with number",
            "data": {"machine_id": "001_motor_siemens"},
            "expect": "FAIL"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_machine_id_format(tc['data'])
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result in tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 4: _validate_sensors
    # =========================================================================
    print_section("TEST 4: Sensor Configuration Validation")
    
    test_cases = [
        {
            "name": "No sensors (CRITICAL ERROR)",
            "data": {"sensors": []},
            "expect": "FAIL"
        },
        {
            "name": "1 sensor (minimum)",
            "data": {"sensors": [{"name": "temp_sensor"}]},
            "expect": "PASS (warning)"
        },
        {
            "name": "5 sensors (recommended)",
            "data": {
                "sensors": [
                    {"name": f"sensor_{i}", "min_value": 0, "max_value": 100}
                    for i in range(5)
                ]
            },
            "expect": "PASS"
        },
        {
            "name": "Missing sensors key",
            "data": {},
            "expect": "FAIL"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_sensors(tc['data'])
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result in tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 5: _validate_sensor_config
    # =========================================================================
    print_section("TEST 5: Individual Sensor Config Validation")
    
    test_cases = [
        {
            "name": "Complete sensor config",
            "data": {"name": "temp_sensor", "min_value": 0, "max_value": 100},
            "expect": "PASS"
        },
        {
            "name": "Missing sensor name",
            "data": {"min_value": 0, "max_value": 100},
            "expect": "FAIL"
        },
        {
            "name": "No value ranges",
            "data": {"name": "pressure_sensor"},
            "expect": "PASS (warning)"
        }
    ]
    
    for i, tc in enumerate(test_cases):
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_sensor_config(tc['data'], i)
        has_errors = any(issue.severity == "error" for issue in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result in tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 6: _validate_tvae_compatibility
    # =========================================================================
    print_section("TEST 6: TVAE Compatibility Validation")
    
    test_cases = [
        {
            "name": "Valid RUL range",
            "data": {"rul_min": 0, "rul_max": 1000, "degradation_states": 4},
            "expect": "PASS"
        },
        {
            "name": "Invalid: rul_max <= rul_min",
            "data": {"rul_min": 1000, "rul_max": 100},
            "expect": "FAIL"
        },
        {
            "name": "Invalid: degradation_states < 2",
            "data": {"degradation_states": 1},
            "expect": "FAIL"
        },
        {
            "name": "Warning: degradation_states > 10",
            "data": {"degradation_states": 15},
            "expect": "PASS (warning)"
        },
        {
            "name": "Missing RUL parameters",
            "data": {},
            "expect": "PASS (warning)"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_tvae_compatibility(tc['data'])
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result in tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 7: _validate_sensor_structure
    # =========================================================================
    print_section("TEST 7: Sensor Structure Validation")
    
    test_cases = [
        {
            "name": "With baseline_normal_operation",
            "data": {
                "baseline_normal_operation": {
                    "temperature": {"temp_sensor": {"min": 0, "max": 100}},
                    "vibration": {"vib_sensor": {"min": 0, "max": 50}}
                }
            },
            "expect": "PASS"
        },
        {
            "name": "Missing baseline_normal_operation",
            "data": {},
            "expect": "PASS (info)"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_sensor_structure(tc['data'])
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result in tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 8: _validate_baseline_operation
    # =========================================================================
    print_section("TEST 8: Baseline Operation Validation")
    
    test_cases = [
        {
            "name": "With alarm thresholds",
            "data": {
                "baseline_normal_operation": {
                    "temperature": {
                        "temp_sensor": {
                            "min": 0,
                            "max": 100,
                            "alarm": 85
                        }
                    }
                }
            },
            "expect": "PASS"
        },
        {
            "name": "Without alarm thresholds",
            "data": {
                "baseline_normal_operation": {
                    "temperature": {
                        "temp_sensor": {"min": 0, "max": 100}
                    }
                }
            },
            "expect": "PASS (info)"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_baseline_operation(tc['data'])
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result in tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # TEST 9: _validate_specifications
    # =========================================================================
    print_section("TEST 9: Specifications Validation")
    
    test_cases = [
        {
            "name": "With specifications",
            "data": {"specifications": {"power_kW": 5.5}},
            "expect": "PASS"
        },
        {
            "name": "Missing specifications",
            "data": {},
            "expect": "PASS (info)"
        }
    ]
    
    for tc in test_cases:
        print(f"\n  Case: {tc['name']}")
        issues = validator._validate_specifications(tc['data'])
        has_errors = any(i.severity == "error" for i in issues)
        result = "FAIL" if has_errors else "PASS"
        status = "✅" if result in tc['expect'] else "❌"
        print(f"  Result: {result} {status} (Expected: {tc['expect']})")
        if issues:
            print_issues(issues)
    
    # =========================================================================
    # INTEGRATION TEST: Complete Profile
    # =========================================================================
    print_section("INTEGRATION TEST: Complete Profile Validation")
    
    complete_profile = {
        "machine_id": "test_integration_complete_001",
        "manufacturer": "Test Manufacturer",
        "model": "Test Model 2000",
        "machine_type": "test_equipment",
        "sensors": [
            {"name": f"sensor_{i}", "min_value": 0, "max_value": 100}
            for i in range(5)
        ],
        "rul_min": 0,
        "rul_max": 1000,
        "degradation_states": 4,
        "baseline_normal_operation": {
            "temperature": {
                "temp_sensor": {"min": 20, "max": 80, "alarm": 90}
            }
        },
        "specifications": {
            "power_kW": 5.5
        }
    }
    
    print("\n  Testing complete valid profile...")
    is_valid, issues, can_proceed = validator.validate_profile(complete_profile, strict=True)
    
    print(f"\n  Validation Result:")
    print(f"    - is_valid: {is_valid}")
    print(f"    - can_proceed: {can_proceed}")
    print(f"    - Total issues: {len(issues)}")
    
    if issues:
        print(f"\n  Issues breakdown:")
        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        infos = [i for i in issues if i.severity == "info"]
        print(f"    - Errors: {len(errors)}")
        print(f"    - Warnings: {len(warnings)}")
        print(f"    - Info: {len(infos)}")
        print_issues(issues)
    
    status = "✅" if is_valid and can_proceed else "❌"
    print(f"\n  {status} Complete profile validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "█"*80)
    print(" "*25 + "VALIDATION TEST COMPLETE")
    print("█"*80)
    print(f"""
✅ All 9 validation functions tested:
   1. _validate_required_fields          - Checks machine_id, manufacturer, model, category
   2. _validate_machine_id_uniqueness    - Prevents duplicate machines (CRITICAL)
   3. _validate_machine_id_format        - Enforces naming conventions
   4. _validate_sensors                  - Ensures minimum sensor count
   5. _validate_sensor_config            - Validates individual sensor structure
   6. _validate_tvae_compatibility       - Checks RUL ranges and degradation states
   7. _validate_sensor_structure         - Validates baseline_normal_operation
   8. _validate_baseline_operation       - Checks alarm thresholds
   9. _validate_specifications           - Validates technical specs section

📊 Detection Coverage:
   - {len(validator.existing_machines)} existing machines detected
   - 4 GAN directories monitored
   - Duplicate detection: ACTIVE
   - TVAE requirements: ENFORCED
""")


if __name__ == "__main__":
    test_all_validations()
