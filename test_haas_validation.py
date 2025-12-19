"""
Direct test of validator with cnc_haas_vf3_001
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'frontend' / 'server'))

from api.services.profile_validator import MachineProfileValidator

# Initialize validator
validator = MachineProfileValidator()

print(f"Total existing machines: {len(validator.existing_machines)}")
print(f"\nChecking if cnc_haas_vf3_001 exists:")
print(f"  In existing_machines: {'cnc_haas_vf3_001' in validator.existing_machines}")
print(f"  In existing_machines (uppercase): {'CNC_HAAS_VF3_001' in validator.existing_machines}")

# List all haas machines
haas_machines = [m for m in validator.existing_machines if 'haas' in m.lower()]
print(f"\nAll Haas machines in system: {haas_machines}")

# Test validation
test_profile = {
    "machine_id": "cnc_haas_vf3_001",
    "manufacturer": "Haas",
    "model": "VF-3",
    "category": "cnc",
    "sensors": [{"name": "temp", "min_value": 0, "max_value": 100}]
}

is_valid, issues, can_proceed = validator.validate_profile(test_profile, strict=True)

print(f"\nValidation Result for cnc_haas_vf3_001:")
print(f"  Valid: {is_valid}")
print(f"  Can Proceed: {can_proceed}")
print(f"\nIssues:")
for issue in issues:
    print(f"  [{issue.severity.upper()}] {issue.field}: {issue.message}")

print(f"\n{'='*60}")
print(f"EXPECTED: FAIL (duplicate)")
print(f"ACTUAL: {'FAIL' if not is_valid else 'PASS'}")
print(f"BUG: {'YES - VALIDATION NOT WORKING!' if is_valid else 'NO - Working correctly'}")
