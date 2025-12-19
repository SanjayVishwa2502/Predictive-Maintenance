"""\
Verify the 14 parsing fallback/normalization rules from `machine_profile_template (1).json`.

This is a lightweight, repo-local test script (not pytest) to keep consistency
with existing `test_*.py` scripts.

Run:
  python test_profile_parsing_fallback_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "frontend" / "server"))

from api.services.profile_validator import MachineProfileValidator  # noqa: E402


def _new_validator() -> MachineProfileValidator:
    return MachineProfileValidator()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_rule_11_alternative_field_names() -> None:
    validator = _new_validator()
    unique = f"unit_test_alt_name_{uuid4().hex[:8]}"

    profile = {
        "name": unique,
        "manufacturer": "Atlas Copco",
        "model": "GA 30",
        "machine_type": "compressor",
        "specs": {"working_pressure_bar": 8.5},
        "baseline_normal_operation": {
            "temperature": {"bearing_temp_C": {"typical": 55}},
        },
    }

    is_valid, issues, _ = validator.validate_profile(profile, strict=False)

    _assert(profile.get("machine_id") == unique, "Rule 11: name -> machine_id mapping failed")
    _assert("specifications" in profile and isinstance(profile["specifications"], dict), "Rule 11: specs -> specifications mapping failed")
    _assert(any(i.field == "machine_id" and "name" in i.message for i in issues), "Rule 11: expected info issue for name mapping")
    _assert(is_valid, "Profile should validate after alternative field mapping")


def test_rule_1_generate_machine_id() -> None:
    validator = _new_validator()

    profile = {
        "manufacturer": "DMG MORI",
        "model": "NLX 2500",
        "category": "cnc",
        "baseline_normal_operation": {
            "mechanical": {"speed_rpm": {"typical": 1500}},
        },
    }

    is_valid, issues, _ = validator.validate_profile(profile, strict=False)

    mid = str(profile.get("machine_id") or "")
    _assert(mid.endswith("_001"), "Rule 1: generated machine_id should end with _001")
    _assert("dmg_mori" in mid, "Rule 1: generated machine_id should include manufacturer token")
    _assert(any(i.field == "machine_id" and "Generated" in i.message for i in issues), "Rule 1: expected info issue for machine_id generation")
    _assert(is_valid, "Profile should validate after machine_id generation")


def test_rule_2_infer_category_from_machine_id() -> None:
    validator = _new_validator()
    unique = f"motor_test_infer_{uuid4().hex[:6]}_001"

    profile = {
        "machine_id": unique,
        "manufacturer": "Siemens",
        "model": "1LA7",
        "machine_type": "motor",
        # category intentionally missing
        "baseline_normal_operation": {"electrical": {"current_A": {"typical": 10}}},
    }

    is_valid, issues, _ = validator.validate_profile(profile, strict=False)

    _assert(profile.get("category") == "motor", "Rule 2: expected category inferred as machine_id prefix")
    _assert(any(i.field == "category" and "Inferred" in i.message for i in issues), "Rule 2: expected info issue for category inference")
    _assert(is_valid, "Profile should validate with inferred category")


def test_rule_3_baseline_alt_field_mapping() -> None:
    validator = _new_validator()
    unique = f"unit_test_baseline_alt_{uuid4().hex[:8]}"

    profile = {
        "machine_id": unique,
        "manufacturer": "Grundfos",
        "model": "CR3-19",
        "machine_type": "pump",
        "operating_parameters": {
            "pressure": {"discharge_pressure_bar": {"typical": 8.5}},
        },
    }

    is_valid, _, _ = validator.validate_profile(profile, strict=False)

    _assert("baseline_normal_operation" in profile, "Rule 3: operating_parameters -> baseline_normal_operation mapping failed")
    _assert(is_valid, "Profile should validate with baseline field fallback")


def test_rule_4_5_9_generate_ranges_and_units() -> None:
    validator = _new_validator()
    unique = f"unit_test_ranges_{uuid4().hex[:8]}"

    profile = {
        "machine_id": unique,
        "manufacturer": "TestCo",
        "model": "X1",
        "machine_type": "motor",
        "baseline_normal_operation": {
            "temperature": {
                "bearing_temp_C": {"typical": 55},  # only typical
            },
            "vibration": {
                "overall_rms_mm_s": 1.2,  # single numeric value
            },
        },
    }

    is_valid, _, _ = validator.validate_profile(profile, strict=False)

    temp = profile["baseline_normal_operation"]["temperature"]["bearing_temp_C"]
    vib = profile["baseline_normal_operation"]["vibration"]["overall_rms_mm_s"]

    _assert("min" in temp and "max" in temp, "Rule 4: min/max not generated from typical")
    _assert(temp.get("unit") == "°C", "Rule 9: unit inference failed for temperature")

    _assert(isinstance(vib, dict) and "typical" in vib, "Rule 5: single value not converted to typical config")
    _assert(vib.get("unit") == "mm/s", "Rule 9: unit inference failed for vibration")

    _assert(is_valid, "Profile should validate after generating ranges/units")


def test_rule_6_7_8_flat_baseline_grouping() -> None:
    validator = _new_validator()
    unique = f"unit_test_flat_baseline_{uuid4().hex[:8]}"

    # baseline_normal_operation is flat: sensor_name -> config
    profile = {
        "machine_id": unique,
        "manufacturer": "TestCo",
        "model": "X2",
        "machine_type": "cnc",
        "baseline_normal_operation": {
            "bearing_temp_C": {"typical": 55},
            "overall_rms_mm_s": {"typical": 1.2},
            "current_A": {"typical": 10},
        },
    }

    is_valid, _, _ = validator.validate_profile(profile, strict=False)
    baseline = profile["baseline_normal_operation"]

    _assert("temperature" in baseline and "bearing_temp_C" in baseline["temperature"], "Rule 6: temperature extraction/grouping failed")
    _assert("vibration" in baseline and "overall_rms_mm_s" in baseline["vibration"], "Rule 7: vibration extraction/grouping failed")
    _assert("electrical" in baseline and "current_A" in baseline["electrical"], "Rule 8: electrical extraction/grouping failed")

    _assert(is_valid, "Profile should validate after flat baseline grouping")


def test_rule_10_manufacturer_spaces_preserved() -> None:
    validator = _new_validator()
    unique = f"unit_test_mfg_space_{uuid4().hex[:8]}_001"

    profile = {
        "machine_id": unique,
        "manufacturer": "DMG MORI",
        "model": "NLX 2500",
        "machine_type": "cnc",
        "baseline_normal_operation": {"mechanical": {"speed_rpm": {"typical": 1500}}},
    }

    is_valid, _, _ = validator.validate_profile(profile, strict=False)

    _assert(profile.get("manufacturer") == "DMG MORI", "Rule 10: manufacturer spacing was modified")
    _assert(is_valid, "Profile should validate with spaced manufacturer")


def test_rule_12_flat_json_top_level_grouping() -> None:
    validator = _new_validator()

    profile = {
        "manufacturer": "TestCo",
        "model": "Flat1",
        "machine_type": "motor",
        # machine_id intentionally missing to also exercise rule 1
        "bearing_temp_C": 55,
        "overall_rms_mm_s": 1.2,
        "current_A": 10,
    }

    is_valid, _, _ = validator.validate_profile(profile, strict=False)

    _assert("baseline_normal_operation" in profile, "Rule 12: expected baseline_normal_operation to be created from flat JSON")
    baseline = profile["baseline_normal_operation"]
    _assert("temperature" in baseline, "Rule 12: missing temperature category in grouped baseline")
    _assert("vibration" in baseline, "Rule 12: missing vibration category in grouped baseline")
    _assert("electrical" in baseline, "Rule 12: missing electrical category in grouped baseline")

    _assert(is_valid, "Profile should validate after flat JSON grouping")


def test_rule_13_fault_signatures_optional() -> None:
    validator = _new_validator()
    unique = f"unit_test_no_faults_{uuid4().hex[:8]}_001"

    profile = {
        "machine_id": unique,
        "manufacturer": "TestCo",
        "model": "NoFaults",
        "machine_type": "motor",
        "baseline_normal_operation": {"electrical": {"current_A": {"typical": 10}}},
        # no fault_signatures
    }

    is_valid, issues, _ = validator.validate_profile(profile, strict=False)

    _assert(not any(i.field == "fault_signatures" and i.severity == "error" for i in issues), "Rule 13: fault_signatures should be optional")
    _assert(is_valid, "Profile should validate without fault_signatures")


def test_rule_14_multiple_formats_normalization() -> None:
    validator = _new_validator()

    profile = {
        "manufacturer": "TestCo",
        "model": "FmtX",
        "category": "motor",
        "sensor_data": {
            "bearing_temp_C": {"typical": 55},
            "overall_rms_mm_s": {"typical": 1.2},
            "current_A": {"typical": 10},
        },
    }

    is_valid, _, _ = validator.validate_profile(profile, strict=False)

    _assert("baseline_normal_operation" in profile, "Rule 14: expected sensor_data to normalize into baseline_normal_operation")
    baseline = profile["baseline_normal_operation"]
    _assert("temperature" in baseline and "bearing_temp_C" in baseline["temperature"], "Rule 14: temperature sensor missing after normalization")
    _assert(is_valid, "Profile should validate after multi-format normalization")


def main() -> None:
    tests = [
        test_rule_11_alternative_field_names,
        test_rule_1_generate_machine_id,
        test_rule_2_infer_category_from_machine_id,
        test_rule_3_baseline_alt_field_mapping,
        test_rule_4_5_9_generate_ranges_and_units,
        test_rule_6_7_8_flat_baseline_grouping,
        test_rule_10_manufacturer_spaces_preserved,
        test_rule_12_flat_json_top_level_grouping,
        test_rule_13_fault_signatures_optional,
        test_rule_14_multiple_formats_normalization,
    ]

    print("=" * 80)
    print("PARSING FALLBACK RULES TEST")
    print("=" * 80)

    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"[PASS] {name}")
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            raise

    print("=" * 80)
    print("ALL RULE TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
