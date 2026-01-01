"""
Machine Profile Validator
Comprehensive validation for machine profiles before creation.
Ensures compliance with template structure and TVAE requirements.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime
import re
import math


def _to_snake_token(value: str) -> str:
    s = str(value or "").strip().lower()
    # keep alnum; convert all other runs to single underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _looks_like_sensor_config(value: Any) -> bool:
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        keys = set(value.keys())
        return bool(keys.intersection({"min", "max", "typical", "alarm", "trip", "unit"}))
    return False


def _infer_sensor_category(sensor_name: str) -> str:
    name = (sensor_name or "").lower()
    if any(k in name for k in ("temp", "temperature")) or name.endswith("_c") or name.endswith("_f"):
        return "temperature"
    if any(k in name for k in ("vibration", "vib", "rms", "mm_s", "mm/s")):
        return "vibration"
    if any(k in name for k in ("current", "_a", "amp")):
        return "electrical"
    if any(k in name for k in ("voltage", "_v", "volt")):
        return "electrical"
    if any(k in name for k in ("power", "_kw", "_mw", "_w")):
        return "electrical"
    if any(k in name for k in ("pressure", "_bar", "psi", "kpa", "mpa")):
        return "pressure"
    if any(k in name for k in ("speed", "rpm", "torque", "nm", "flow", "m3_h")):
        return "mechanical"
    if any(k in name for k in ("sound", "acoustic", "dba")):
        return "acoustic"
    return "other"


def _infer_unit(sensor_name: str) -> Optional[str]:
    name = (sensor_name or "").lower()
    # note: keep this intentionally small and predictable
    if "rpm" in name or name.endswith("_rpm"):
        return "rpm"
    if "_kw" in name or name.endswith("kw") or "kilowatt" in name:
        return "kW"
    if "_bar" in name or "bar" in name:
        return "bar"
    if "temp" in name or "temperature" in name or name.endswith("_c"):
        return "°C"
    if any(k in name for k in ("mm_s", "mm/s", "rms", "vibration", "vib")):
        return "mm/s"
    if "current" in name or name.endswith("_a"):
        return "A"
    if "voltage" in name or name.endswith("_v"):
        return "V"
    if "dba" in name:
        return "dBA"
    if name.endswith("_nm") or "torque" in name:
        return "Nm"
    if "m3_h" in name or "flow" in name:
        return "m³/h"
    return None


class ValidationIssue:
    """Validation issue details"""
    def __init__(self, severity: str, field: str, message: str):
        self.severity = severity  # "error", "warning", "info"
        self.field = field
        self.message = message
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "severity": self.severity,
            "field": self.field,
            "message": self.message
        }


class MachineProfileValidator:
    """
    Comprehensive machine profile validator.
    
    Validation Rules:
    1. Machine ID uniqueness (check existing machines)
    2. Required fields present (machine_id, manufacturer, model, category)
    3. Sensor configuration valid (minimum count, structure)
    4. TVAE compatibility (numeric sensors, proper ranges)
    5. Profile structure compliance with template
    """
    
    def __init__(self, gan_metadata_dir: Path = None):
        """
        Initialize validator.
        
        # Check 5: GAN/seed_data (seed artifacts can exist before profiles/metadata)
        seed_dir = self.project_root / "GAN" / "seed_data"
        if seed_dir.exists():
            # Common patterns: <machine_id>_seed.parquet, <machine_id>_temporal_seed.parquet
            for seed_file in seed_dir.rglob("*.parquet"):
                name = seed_file.stem
                for suffix in ("_seed", "_temporal_seed"):
                    if name.endswith(suffix):
                        existing.add(name[: -len(suffix)])
                        break
        Args:
            gan_metadata_dir: Path to GAN metadata directory
        """
        self.project_root = self._find_project_root()
        self.gan_metadata_dir = gan_metadata_dir or (self.project_root / "GAN" / "metadata")
        self.template_path = self.project_root / "machine_profile_template (1).json"

        self.allowed_machine_types: Set[str] = {
            "cnc",
            "motor",
            "induction_motor",
            "pump",
            "compressor",
            "fan",
            "conveyor",
            "hydraulic",
            "robot",
            "transformer",
            "cooling",
            "turbofan",
            "test_equipment",
        }

        self.allowed_sensor_types: Set[str] = {
            "temperature",
            "vibration",
            "pressure",
            "electrical",
            "current",
            "voltage",
            "power",
            "speed",
            "flow",
            "position",
            "torque",
            "acoustic",
            "other",
        }
        
        # Load existing machines
        self.existing_machines = self._load_existing_machines()
        self.pending_machine_ids = self._load_pending_machine_ids()
        
        # TVAE requirements
        self.min_sensors = 1
        self.recommended_sensors = 5
        self.max_sensors = 50
    
    def _load_existing_machines(self) -> set:
        """
        Load all existing machine IDs from all GAN directories.
        Checks multiple locations to ensure comprehensive duplicate detection.
        """
        existing = set()

        # Check 1: GAN/metadata directory (SDV metadata files)
        metadata_dir = self.gan_metadata_dir
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                if metadata_file.stem.endswith('_metadata'):
                    # SDV metadata format: extract from filename
                    machine_id = metadata_file.stem.replace('_metadata', '')
                    existing.add(machine_id)
                elif not metadata_file.stem.endswith('_profile_temp'):
                    # Regular profile format: extract from filename
                    machine_id = metadata_file.stem
                    existing.add(machine_id)
        
        # Check 2: GAN/data/real_machines/profiles directory (machine profiles)
        real_machines_dir = self.project_root / "GAN" / "data" / "real_machines" / "profiles"
        if real_machines_dir.exists():
            for profile_file in real_machines_dir.glob("*.json"):
                machine_id = profile_file.stem
                existing.add(machine_id)
        
        # Check 3: GAN/data/synthetic directory (machines with generated data)
        synthetic_dir = self.project_root / "GAN" / "data" / "synthetic"
        if synthetic_dir.exists():
            for machine_dir in synthetic_dir.iterdir():
                if machine_dir.is_dir():
                    machine_id = machine_dir.name.replace('_synthetic_temporal', '')
                    existing.add(machine_id)
        
        # Check 4: GAN/models directory (machines with trained models)
        models_dir = self.project_root / "GAN" / "models"
        if models_dir.exists():
            for model_dir in models_dir.iterdir():
                if model_dir.is_dir() and not model_dir.name.startswith('.'):
                    existing.add(model_dir.name)
        
        return existing

    def _load_pending_machine_ids(self) -> set:
        """Machine IDs that currently have a staged (temp) profile upload."""
        pending = set()
        metadata_dir = self.gan_metadata_dir
        if not metadata_dir.exists():
            return pending

        # Primary mechanism: machine_id_profile_temp.json (fast existence + parse-free)
        for temp_file in metadata_dir.glob("*_profile_temp.json"):
            stem = temp_file.stem
            # profile_id_profile_temp.json doesn't encode machine id; try to read minimal json
            if stem.endswith("_profile_temp"):
                maybe_machine_id = stem.replace("_profile_temp", "")
                if maybe_machine_id and "_" in maybe_machine_id:
                    pending.add(maybe_machine_id.lower())
                    continue

            # Fallback: load and use embedded machine_id
            try:
                import json
                with open(temp_file, "r") as f:
                    data = json.load(f)
                mid = str(data.get("machine_id", "")).strip().lower()
                if mid:
                    pending.add(mid)
            except Exception:
                continue

        return pending

    def is_pending_upload(self, machine_id: str) -> bool:
        return (machine_id or "").strip().lower() in self.pending_machine_ids

    def _find_project_root(self) -> Path:
        """Resolve project root reliably regardless of process CWD."""
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "GAN").exists() and (parent / "frontend").exists():
                return parent
        return Path.cwd().resolve()

    def _normalize_machine_type(self, profile_data: Dict[str, Any]) -> Optional[str]:
        machine_type = (profile_data.get("machine_type") or "").strip().lower()
        category = (profile_data.get("category") or "").strip().lower()
        if machine_type:
            return machine_type
        if category:
            # template uses human-friendly category; also allow passing short machine type as category
            return category
        return None

    def _extract_sensors_any_format(self, profile_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sensors = profile_data.get("sensors")
        if isinstance(sensors, list):
            return sensors
        if isinstance(sensors, dict):
            return self._extract_sensors_from_baseline(sensors)

        baseline = profile_data.get("baseline_normal_operation")
        if isinstance(baseline, dict):
            return self._extract_sensors_from_baseline(baseline)

        return []

    def _apply_parsing_fallbacks(self, profile_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Apply the template's parsing/normalization rules (rule_1..rule_14).

        This mutates profile_data in-place so that subsequent validations run against
        the normalized structure.
        """
        issues: List[ValidationIssue] = []

        if not isinstance(profile_data, dict):
            return issues

        # Rule 11: alternative field names
        if not profile_data.get("machine_id") and profile_data.get("name"):
            profile_data["machine_id"] = str(profile_data.get("name")).strip()
            issues.append(
                ValidationIssue(
                    severity="info",
                    field="machine_id",
                    message="Mapped alternative field 'name' -> 'machine_id'.",
                )
            )

        if not profile_data.get("specifications") and profile_data.get("specs"):
            profile_data["specifications"] = profile_data.get("specs")
            issues.append(
                ValidationIssue(
                    severity="info",
                    field="specifications",
                    message="Mapped alternative field 'specs' -> 'specifications'.",
                )
            )

        # Rule 2: infer category if missing
        if not profile_data.get("category"):
            machine_id = str(profile_data.get("machine_id", "") or "").strip().lower()
            inferred: Optional[str] = None
            if machine_id and "_" in machine_id:
                inferred = machine_id.split("_")[0]
            else:
                specs = profile_data.get("specifications")
                if isinstance(specs, dict):
                    spec_keys = " ".join([str(k).lower() for k in specs.keys()])
                    if any(k in spec_keys for k in ("spindle", "tool_positions", "turning", "machining")):
                        inferred = "cnc"
                    elif any(k in spec_keys for k in ("voltage", "current", "frequency", "poles", "rated_speed")):
                        inferred = "motor"
                    elif any(k in spec_keys for k in ("flow", "head", "stages", "impeller")):
                        inferred = "pump"
                    elif any(k in spec_keys for k in ("free_air", "working_pressure", "oil_capacity", "compressor")):
                        inferred = "compressor"

            if inferred:
                profile_data["category"] = inferred
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field="category",
                        message=f"Inferred missing category as '{inferred}'.",
                    )
                )

        # Rule 1: generate machine_id if missing
        if not profile_data.get("machine_id"):
            category = profile_data.get("category") or profile_data.get("machine_type")
            manufacturer = profile_data.get("manufacturer")
            model = profile_data.get("model")
            if category and manufacturer and model:
                generated = f"{_to_snake_token(category)}_{_to_snake_token(manufacturer)}_{_to_snake_token(model)}_001"
                profile_data["machine_id"] = generated
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field="machine_id",
                        message=f"Generated missing machine_id as '{generated}'.",
                    )
                )

        # Rule 10: manufacturer with spaces should remain intact.
        # Intentionally do not normalize manufacturer/model fields.

        # Rule 3: baseline_normal_operation fallbacks
        if not profile_data.get("baseline_normal_operation"):
            for alt in ("operating_parameters", "sensor_data", "normal_conditions"):
                candidate = profile_data.get(alt)
                if isinstance(candidate, dict):
                    profile_data["baseline_normal_operation"] = candidate
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            field="baseline_normal_operation",
                            message=f"Mapped alternative baseline field '{alt}' -> 'baseline_normal_operation'.",
                        )
                    )
                    break

        # Rule 12/14: if JSON is flat or baseline is flat, auto-group sensors into baseline_normal_operation
        baseline = profile_data.get("baseline_normal_operation")
        if isinstance(baseline, dict):
            # Detect flat baseline: a dict of sensor_name -> config/value rather than category -> sensors
            non_comment_items = {k: v for k, v in baseline.items() if isinstance(k, str) and not k.startswith("_")}
            looks_nested = any(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values()) for v in non_comment_items.values())
            if non_comment_items and not looks_nested:
                grouped: Dict[str, Dict[str, Any]] = {}
                for sensor_name, sensor_value in non_comment_items.items():
                    if not _looks_like_sensor_config(sensor_value):
                        continue
                    category = _infer_sensor_category(sensor_name)
                    grouped.setdefault(category, {})[sensor_name] = (
                        {"typical": sensor_value} if isinstance(sensor_value, (int, float)) else sensor_value
                    )
                if grouped:
                    profile_data["baseline_normal_operation"] = grouped
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            field="baseline_normal_operation",
                            message="Detected flat baseline and auto-grouped sensors into categories.",
                        )
                    )

        # If still no baseline and no sensors list, attempt flat top-level grouping
        if not profile_data.get("baseline_normal_operation") and not profile_data.get("sensors"):
            meta_keys = {
                "machine_id",
                "manufacturer",
                "model",
                "category",
                "machine_type",
                "application",
                "data_source",
                "specifications",
                "fault_signatures",
                "maintenance_schedule",
                "validation_data",
                "notes",
                "profile_id",
            }
            grouped: Dict[str, Dict[str, Any]] = {}
            for key, value in profile_data.items():
                if key in meta_keys or (isinstance(key, str) and key.startswith("_")):
                    continue
                if not isinstance(key, str):
                    continue
                if _looks_like_sensor_config(value):
                    category = _infer_sensor_category(key)
                    grouped.setdefault(category, {})[key] = (
                        {"typical": value} if isinstance(value, (int, float)) else value
                    )
            if grouped:
                profile_data["baseline_normal_operation"] = grouped
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field="baseline_normal_operation",
                        message="Detected flat JSON input and auto-grouped sensor fields into baseline_normal_operation.",
                    )
                )

        # Rule 4/5/6/7/8/9: fill in sensor configs (ranges + units) inside baseline
        baseline = profile_data.get("baseline_normal_operation")
        if isinstance(baseline, dict):
            for category, sensors_dict in list(baseline.items()):
                if not isinstance(category, str) or category.startswith("_"):
                    continue
                if not isinstance(sensors_dict, dict):
                    continue
                for sensor_name, sensor_config in list(sensors_dict.items()):
                    if not isinstance(sensor_name, str) or sensor_name.startswith("_"):
                        continue

                    # support single numeric value
                    if isinstance(sensor_config, (int, float)):
                        sensors_dict[sensor_name] = {"typical": sensor_config}
                        sensor_config = sensors_dict[sensor_name]
                        issues.append(
                            ValidationIssue(
                                severity="info",
                                field=f"baseline_normal_operation.{category}.{sensor_name}",
                                message="Converted single sensor value to config with 'typical'.",
                            )
                        )

                    if not isinstance(sensor_config, dict):
                        continue

                    # if only typical, generate min/max
                    if "typical" in sensor_config and ("min" not in sensor_config or "max" not in sensor_config):
                        try:
                            typical = float(sensor_config.get("typical"))
                            span = abs(typical) * 0.2
                            if span == 0:
                                span = 0.2
                            sensor_config.setdefault("min", typical - span)
                            sensor_config.setdefault("max", typical + span)
                            issues.append(
                                ValidationIssue(
                                    severity="info",
                                    field=f"baseline_normal_operation.{category}.{sensor_name}",
                                    message="Auto-generated min/max as ±20% of typical.",
                                )
                            )
                        except Exception:
                            pass

                    # if min/max present but typical missing, infer midpoint
                    if "typical" not in sensor_config and "min" in sensor_config and "max" in sensor_config:
                        try:
                            mn = float(sensor_config.get("min"))
                            mx = float(sensor_config.get("max"))
                            sensor_config["typical"] = (mn + mx) / 2.0
                            issues.append(
                                ValidationIssue(
                                    severity="info",
                                    field=f"baseline_normal_operation.{category}.{sensor_name}",
                                    message="Inferred missing typical as midpoint of min/max.",
                                )
                            )
                        except Exception:
                            pass

                    # infer unit
                    if not sensor_config.get("unit"):
                        inferred_unit = _infer_unit(sensor_name)
                        if inferred_unit:
                            sensor_config["unit"] = inferred_unit
                            issues.append(
                                ValidationIssue(
                                    severity="info",
                                    field=f"baseline_normal_operation.{category}.{sensor_name}.unit",
                                    message=f"Inferred missing unit as '{inferred_unit}'.",
                                )
                            )

        return issues
    
    def validate_profile(self, profile_data: Dict[str, Any], strict: bool = True) -> Tuple[bool, List[ValidationIssue], bool]:
        """
        Validate machine profile comprehensively.
        
        Args:
            profile_data: Machine profile dictionary
            strict: Enable strict validation (recommended)
        
        Returns:
            Tuple of (is_valid, issues_list, can_proceed)
            - is_valid: True if no errors found
            - issues_list: List of ValidationIssue objects
            - can_proceed: True if machine can be created (no blocking errors)
        """
        issues: List[ValidationIssue] = []

        # Apply intelligent parsing/normalization rules (template _parsing_fallback_rules)
        issues.extend(self._apply_parsing_fallbacks(profile_data))
        
        # CRITICAL VALIDATIONS (blocking errors)
        issues.extend(self._validate_required_fields(profile_data))
        issues.extend(self._validate_machine_id_uniqueness(profile_data, strict))
        issues.extend(self._validate_machine_id_format(profile_data))
        issues.extend(self._validate_machine_type_consistency(profile_data, strict))
        issues.extend(self._validate_sensors(profile_data, strict))
        issues.extend(self._validate_sensor_names_and_uniqueness(profile_data))
        issues.extend(self._validate_sensor_ranges_and_units(profile_data, strict))
        issues.extend(self._validate_tvae_compatibility(profile_data, strict))
        
        # RECOMMENDED VALIDATIONS (warnings)
        issues.extend(self._validate_sensor_structure(profile_data))
        issues.extend(self._validate_baseline_operation(profile_data))
        issues.extend(self._validate_specifications(profile_data))
        
        # Determine validation result
        has_errors = any(issue.severity == "error" for issue in issues)
        is_valid = not has_errors
        can_proceed = is_valid
        
        return is_valid, issues, can_proceed

    def _validate_machine_type_consistency(self, profile_data: Dict[str, Any], strict: bool) -> List[ValidationIssue]:
        """Validate machine_type/category present and consistent with machine_id prefix."""
        issues: List[ValidationIssue] = []
        machine_id = (profile_data.get("machine_id") or "").strip().lower()
        machine_type = self._normalize_machine_type(profile_data)

        if not machine_type:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="machine_type",
                    message="machine_type (or category) is required. It must match the machine_id prefix (e.g., cnc_*, motor_*).",
                )
            )
            return issues

        # If machine_type is a long descriptive category (template), don't hard fail; but still try to infer
        inferred_prefix = machine_id.split("_")[0] if machine_id else ""
        if inferred_prefix and machine_type and (machine_type in self.allowed_machine_types):
            # Allow compound types like "test_equipment" to map to "test" prefix
            compound_prefix = machine_type.split("_")[0] if "_" in machine_type else machine_type
            if inferred_prefix != machine_type and inferred_prefix != compound_prefix and inferred_prefix not in machine_type:
                issues.append(
                    ValidationIssue(
                        severity="error" if strict else "warning",
                        field="machine_type",
                        message=f"machine_type '{machine_type}' does not match machine_id prefix '{inferred_prefix}'.",
                    )
                )

        # Basic sanity: discourage very long / sentence-like machine_type
        if len(machine_type) > 60:
            issues.append(
                ValidationIssue(
                    severity="info",
                    field="machine_type",
                    message="machine_type/category is very long. Prefer a short type like 'cnc', 'motor', 'pump' for workflow consistency.",
                )
            )

        # If both present, ensure they are not contradictory
        if profile_data.get("machine_type") and profile_data.get("category"):
            mt = str(profile_data.get("machine_type") or "").strip().lower()
            cat = str(profile_data.get("category") or "").strip().lower()
            if mt and cat and (mt not in cat) and (cat not in mt) and strict:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field="category",
                        message="machine_type and category differ. Ensure they describe the same machine (e.g., machine_type='cnc', category='CNC Vertical Machining Center').",
                    )
                )

        return issues

    def _validate_sensor_names_and_uniqueness(self, profile_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate sensor names exist, follow a safe naming convention, and are unique."""
        issues: List[ValidationIssue] = []
        sensors = self._extract_sensors_any_format(profile_data)

        seen: Set[str] = set()
        for i, sensor in enumerate(sensors):
            name = ""
            if isinstance(sensor, dict):
                name = str(sensor.get("name") or "").strip()
                if not name and "config" in sensor and isinstance(sensor.get("config"), dict):
                    # baseline-extracted sensors have name already; keep fallback in case
                    name = str(sensor.get("sensor_name") or "").strip()

            if not name:
                continue

            normalized = name.lower()
            if normalized in seen:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field=f"sensors[{i}].name",
                        message=f"Duplicate sensor name '{name}'. Sensor names must be unique.",
                    )
                )
            seen.add(normalized)

            # Strongly recommended naming convention for stable feature engineering
            if not re.match(r"^[a-z0-9_]+$", normalized):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        field=f"sensors[{i}].name",
                        message=f"Sensor name '{name}' should be lowercase snake_case (letters/numbers/underscores only) to avoid TVAE feature inconsistencies.",
                    )
                )

        if sensors and len(seen) == 0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="sensors",
                    message="Sensors were provided but none have a valid 'name'.",
                )
            )

        return issues

    def _validate_sensor_ranges_and_units(self, profile_data: Dict[str, Any], strict: bool) -> List[ValidationIssue]:
        """Validate min/max/typical/alarm/trip relationships and unit presence (template + upload formats)."""
        issues: List[ValidationIssue] = []
        sensors = self._extract_sensors_any_format(profile_data)

        for i, sensor in enumerate(sensors):
            if not isinstance(sensor, dict):
                continue

            # Upload format: min_value/max_value/unit/sensor_type
            min_v = sensor.get("min_value")
            max_v = sensor.get("max_value")
            unit = sensor.get("unit")
            s_type = (sensor.get("sensor_type") or sensor.get("type") or sensor.get("category") or "").strip().lower()

            # Baseline-extracted format: config with min/typical/max/alarm/trip/unit
            cfg = sensor.get("config") if isinstance(sensor.get("config"), dict) else sensor
            if min_v is None and isinstance(cfg, dict):
                min_v = cfg.get("min")
            if max_v is None and isinstance(cfg, dict):
                max_v = cfg.get("max")
            if unit is None and isinstance(cfg, dict):
                unit = cfg.get("unit")

            # Unit is useful for normalization/interpretability, but not required to stage a profile.
            if unit in (None, ""):
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field=f"sensors[{i}].unit",
                        message="Sensor unit is missing. Units are strongly recommended for normalization and interpretability.",
                    )
                )

            # Sensor type requirement in upload format
            if "sensor_type" in sensor and s_type and s_type not in self.allowed_sensor_types:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field=f"sensors[{i}].sensor_type",
                        message=f"Unknown sensor_type '{s_type}'. Recommended types: {', '.join(sorted(self.allowed_sensor_types))}.",
                    )
                )

            # Validate numeric range if present
            if min_v is not None and max_v is not None:
                try:
                    min_f = float(min_v)
                    max_f = float(max_v)
                    if max_f <= min_f:
                        issues.append(
                            ValidationIssue(
                                severity="error",
                                field=f"sensors[{i}]",
                                message=f"Invalid sensor range: max ({max_f}) must be > min ({min_f}).",
                            )
                        )
                except Exception:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            field=f"sensors[{i}]",
                            message="Sensor min/max must be numeric.",
                        )
                    )

            # Template relationships if present
            if isinstance(cfg, dict):
                try:
                    c_min = cfg.get("min")
                    c_typ = cfg.get("typical")
                    c_max = cfg.get("max")
                    c_alarm = cfg.get("alarm")
                    c_trip = cfg.get("trip")

                    if c_min is not None and c_max is not None:
                        cmin = float(c_min)
                        cmax = float(c_max)
                        if cmax <= cmin:
                            issues.append(
                                ValidationIssue(
                                    severity="error",
                                    field=f"baseline_normal_operation",
                                    message=f"Baseline sensor has invalid min/max: max ({cmax}) must be > min ({cmin}).",
                                )
                            )
                        if c_typ is not None:
                            ctyp = float(c_typ)
                            if ctyp < cmin or ctyp > cmax:
                                issues.append(
                                    ValidationIssue(
                                        severity="info",
                                        field="baseline_normal_operation",
                                        message="Baseline sensor 'typical' value is outside min/max range.",
                                    )
                                )
                    if c_alarm is not None and c_max is not None:
                        if float(c_alarm) <= float(c_max):
                            issues.append(
                                ValidationIssue(
                                    severity="info",
                                    field="baseline_normal_operation",
                                    message="Baseline sensor 'alarm' should be above the 'max' normal range.",
                                )
                            )
                    if c_trip is not None and c_alarm is not None:
                        if float(c_trip) <= float(c_alarm):
                            issues.append(
                                ValidationIssue(
                                    severity="info",
                                    field="baseline_normal_operation",
                                    message="Baseline sensor 'trip' should be above the 'alarm' threshold.",
                                )
                            )
                except Exception:
                    # Don't hard-fail baseline parsing for non-numeric configs; just warn.
                    issues.append(
                        ValidationIssue(
                            severity="info",
                            field="baseline_normal_operation",
                            message="Some baseline sensor thresholds are non-numeric; TVAE training typically expects numeric ranges.",
                        )
                    )

        # Optional: require at least one critical sensor (upload format)
        sensors_upload = profile_data.get("sensors")
        if isinstance(sensors_upload, list) and sensors_upload:
            has_critical = any(bool(s.get("is_critical")) for s in sensors_upload if isinstance(s, dict))
            if not has_critical:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field="sensors",
                        message="No sensor marked is_critical=true. Consider marking key sensors as critical.",
                    )
                )

        return issues
    
    def _validate_required_fields(self, profile_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate all required fields are present"""
        issues = []
        required_fields = {
            'machine_id': 'Machine ID is required (format: <type>_<manufacturer>_<model>_<id>)',
            'manufacturer': 'Manufacturer name is required',
            'model': 'Model number or designation is required',
        }
        
        for field, message in required_fields.items():
            if not profile_data.get(field):
                issues.append(ValidationIssue(
                    severity="error",
                    field=field,
                    message=message
                ))

        # Support both new upload schema (machine_type) and legacy schema (category)
        machine_type = (profile_data.get('machine_type') or '').strip()
        category = (profile_data.get('category') or '').strip()
        if not machine_type and not category:
            issues.append(
                ValidationIssue(
                    severity="error",
                    field="machine_type",
                    message="machine_type (or legacy category) is required (e.g., motor, pump, cnc)",
                )
            )
        
        return issues
    
    def _validate_machine_id_uniqueness(self, profile_data: Dict[str, Any], strict: bool) -> List[ValidationIssue]:
        """
        CRITICAL: Check if machine_id already exists.
        This prevents duplicate machines and data conflicts.
        """
        issues = []
        machine_id = profile_data.get('machine_id', '').lower()
        
        if not machine_id:
            return issues  # Already caught by required fields check
        
        # Check against existing machines in metadata directory
        if machine_id in self.existing_machines:
            issues.append(ValidationIssue(
                severity="error",
                field="machine_id",
                message=(
                    f"Machine ID '{machine_id}' already exists. "
                    f"Edit 'machine_id' to a new unique value to create a new machine, "
                    f"or use the Existing Machines workflow for the current machine."
                )
            ))
        
        # Additional check: look for similar machine IDs (typo detection)
        if strict:
            similar_machines = self._find_similar_machine_ids(machine_id)
            if similar_machines:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field="machine_id",
                        message=f"Found similar machine IDs: {', '.join(similar_machines)}. "
                        f"Verify this is not a duplicate or typo.",
                    )
                )
        
        return issues
    
    def _find_similar_machine_ids(self, machine_id: str) -> List[str]:
        """Find similar machine IDs (potential duplicates)"""
        similar = []
        parts = machine_id.split('_')
        
        # Check for machines with same type and manufacturer
        if len(parts) >= 2:
            type_prefix = parts[0]
            manufacturer = parts[1] if len(parts) > 1 else ''
            
            for existing_id in self.existing_machines:
                existing_parts = existing_id.split('_')
                if len(existing_parts) >= 2:
                    if existing_parts[0] == type_prefix and existing_parts[1] == manufacturer:
                        similar.append(existing_id)
        
        return similar[:5]  # Return max 5 similar machines
    
    def _validate_machine_id_format(self, profile_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate machine_id format follows convention"""
        issues = []
        machine_id = profile_data.get('machine_id', '')
        
        if not machine_id:
            return issues  # Already caught by required fields
        
        # Check format: <type>_<manufacturer>_<model>_<id>
        pattern = r'^[a-z][a-z0-9_]*[a-z0-9]$'
        if not re.match(pattern, machine_id):
            issues.append(ValidationIssue(
                severity="error",
                field="machine_id",
                message=f"Invalid machine_id format. Must be lowercase with underscores, "
                        f"format: <type>_<manufacturer>_<model>_<id>. "
                        f"Example: motor_siemens_1la7_001"
            ))
        
        # Check for consecutive underscores
        if '__' in machine_id:
            issues.append(ValidationIssue(
                severity="error",
                field="machine_id",
                message="machine_id cannot contain consecutive underscores"
            ))
        
        # Check minimum parts
        parts = machine_id.split('_')
        if len(parts) < 3:
            issues.append(ValidationIssue(
                severity="info",
                field="machine_id",
                message=f"machine_id has only {len(parts)} parts. "
                        f"Recommended format: <type>_<manufacturer>_<model>_<id> (4 parts)"
            ))
        
        return issues
    
    def _validate_sensors(self, profile_data: Dict[str, Any], strict: bool) -> List[ValidationIssue]:
        """Validate sensor configuration"""
        issues = []
        
        # Check sensors field exists
        if 'sensors' not in profile_data:
            # Try to extract from baseline_normal_operation
            if 'baseline_normal_operation' in profile_data:
                issues.append(ValidationIssue(
                    severity="info",
                    field="sensors",
                    message="Sensors will be extracted from baseline_normal_operation structure"
                ))
                return issues
            else:
                issues.append(ValidationIssue(
                    severity="error",
                    field="sensors",
                    message="CRITICAL: No sensors defined. Machine must have at least 1 sensor. "
                            "TVAE requires sensor data for training."
                ))
                return issues
        
        sensors = profile_data.get('sensors', [])
        
        # Convert baseline_normal_operation to sensors list if needed
        if isinstance(sensors, dict):
            sensors = self._extract_sensors_from_baseline(sensors)
        
        # Validate sensor count
        sensor_count = len(sensors) if isinstance(sensors, list) else 0
        
        if sensor_count < self.min_sensors:
            issues.append(ValidationIssue(
                severity="error",
                field="sensors",
                message=f"CRITICAL: At least {self.min_sensors} sensor is required. "
                        f"Current: {sensor_count}. TVAE cannot train without sensor data."
            ))
        elif sensor_count < self.recommended_sensors:
            issues.append(ValidationIssue(
                severity="info",
                field="sensors",
                message=f"Only {sensor_count} sensors configured. "
                        f"Recommended: {self.recommended_sensors}+ sensors for better predictions."
            ))
        
        if sensor_count > self.max_sensors:
            issues.append(ValidationIssue(
                severity="info",
                field="sensors",
                message=f"High sensor count ({sensor_count}). "
                        f"Consider reducing to {self.max_sensors} or less for faster training."
            ))
        
        # Validate individual sensors
        if isinstance(sensors, list) and sensors:
            for i, sensor in enumerate(sensors):
                if isinstance(sensor, dict):
                    issues.extend(self._validate_sensor_config(sensor, i))
        
        return issues
    
    def _extract_sensors_from_baseline(self, baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sensor list from baseline_normal_operation structure"""
        sensors = []
        
        for category, sensors_dict in baseline.items():
            if isinstance(sensors_dict, dict) and not category.startswith('_'):
                for sensor_name, sensor_config in sensors_dict.items():
                    if isinstance(sensor_config, dict) and not sensor_name.startswith('_'):
                        sensors.append({
                            'name': sensor_name,
                            'category': category,
                            'config': sensor_config
                        })
        
        return sensors
    
    def _validate_sensor_config(self, sensor: Dict[str, Any], index: int) -> List[ValidationIssue]:
        """Validate individual sensor configuration"""
        issues = []
        
        # Check sensor name
        if 'name' not in sensor:
            issues.append(ValidationIssue(
                severity="error",
                field=f"sensors[{index}].name",
                message=f"Sensor #{index+1} missing 'name' field"
            ))
        
        # Check sensor has value range or config
        has_values = any(key in sensor for key in ['min_value', 'max_value', 'typical', 'min', 'max'])
        if not has_values:
            issues.append(ValidationIssue(
                severity="info",
                field=f"sensors[{index}]",
                message=f"Sensor '{sensor.get('name', f'#{index+1}')}' has no value range defined"
            ))
        
        return issues
    
    def _validate_tvae_compatibility(self, profile_data: Dict[str, Any], strict: bool) -> List[ValidationIssue]:
        """
        Validate profile compatibility with TVAE (Temporal Variational Autoencoder).
        TVAE requires:
        - Numeric sensor data
        - Proper value ranges (for normalization)
        - RUL (Remaining Useful Life) parameters
        - Temporal consistency
        """
        issues = []
        
        machine_id = (profile_data.get("machine_id") or "").strip().lower()

        # RUL config is not required inside the *profile JSON* for this project.
        # The GAN pipeline defines RUL behavior in GAN/config/rul_profiles.py and bakes RUL into
        # temporal seed data + trained temporal TVAE models.
        has_rul_config = any(key in profile_data for key in ["rul_min", "rul_max", "rul_parameters"])
        
        if not has_rul_config:
            # If this machine already exists, the duplicate error is the only thing that matters.
            # Don't spam additional RUL guidance.
            if machine_id and machine_id in self.existing_machines:
                return issues

            baseline = profile_data.get("baseline_normal_operation", {})
            if "rul_parameters" not in baseline and "degradation_states" not in profile_data:
                issues.append(
                    ValidationIssue(
                        severity="info",
                        field="rul_parameters",
                        message=(
                            "No RUL configuration found in the profile JSON. "
                            "In this project, RUL behavior is defined in GAN/config/rul_profiles.py "
                            "and included during temporal seed generation/training. "
                            "New machines are matched to an RUL profile by category (the machine_id prefix, e.g., 'cnc_*', 'pump_*'). "
                            "If you introduce a new category, add a category entry to GAN/config/rul_profiles.py for realistic degradation."
                        ),
                    )
                )
        
        # Validate RUL range if present.
        # NOTE: Some machines (e.g., simple thermal logs like 3D printers) may not have an RUL label.
        # For those, we allow `rul_min`/`rul_max` to be omitted or set to a placeholder.
        def _parse_optional_rul(value: Any) -> Tuple[Optional[float], bool, bool]:
            """Return (parsed_value, is_missing, is_invalid).

            Missing means: None, "", "null", "none", "nan" (string), or NaN float.
            Invalid means: provided but not numeric.
            """
            if value is None:
                return None, True, False
            if isinstance(value, str):
                s = value.strip().lower()
                if s in ("", "null", "none", "nan"):
                    return None, True, False
            try:
                f = float(value)
                if math.isnan(f):
                    return None, True, False
                return f, False, False
            except Exception:
                return None, False, True

        has_rul_min = "rul_min" in profile_data
        has_rul_max = "rul_max" in profile_data
        if has_rul_min or has_rul_max:
            raw_min = profile_data.get("rul_min")
            raw_max = profile_data.get("rul_max")

            rul_min, min_missing, min_invalid = _parse_optional_rul(raw_min)
            rul_max, max_missing, max_invalid = _parse_optional_rul(raw_max)

            if min_invalid:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field="rul_min",
                        message="rul_min must be numeric, null, or 'NaN' (string) to indicate no RUL.",
                    )
                )
            if max_invalid:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field="rul_max",
                        message="rul_max must be numeric, null, or 'NaN' (string) to indicate no RUL.",
                    )
                )

            # If both are missing (or placeholders), treat as "no RUL" and warn.
            if min_missing and max_missing:
                issues.append(
                    ValidationIssue(
                        severity="warning" if strict else "info",
                        field="rul_range",
                        message=(
                            "No RUL configured for this machine (rul_min/rul_max are missing or set to a placeholder). "
                            "RUL regression training/validation will be skipped."
                        ),
                    )
                )
            # If one side is present and the other is missing => inconsistent.
            elif min_missing != max_missing:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        field="rul_range",
                        message=(
                            "Incomplete RUL range: provide both rul_min and rul_max, or set both to null/'NaN' to indicate no RUL."
                        ),
                    )
                )
            # Both numeric: validate ordering.
            elif rul_min is not None and rul_max is not None:
                if rul_max <= rul_min:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            field="rul_range",
                            message=f"Invalid RUL range: rul_max ({rul_max}) must be greater than rul_min ({rul_min})",
                        )
                    )
        
        # Check degradation states
        if 'degradation_states' in profile_data:
            states = profile_data.get('degradation_states', 0)
            if states < 2:
                issues.append(ValidationIssue(
                    severity="error",
                    field="degradation_states",
                    message=f"degradation_states must be >= 2 (got {states}). "
                            "TVAE requires multiple health states for training."
                ))
            elif states > 10:
                issues.append(ValidationIssue(
                    severity="info" if not strict else "warning",
                    field="degradation_states",
                    message=f"High number of degradation states ({states}). "
                            "Consider using 3-5 states for typical industrial equipment."
                ))
        
        return issues
    
    def _validate_sensor_structure(self, profile_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate sensor data structure in baseline_normal_operation"""
        issues = []
        
        if 'baseline_normal_operation' not in profile_data:
            issues.append(ValidationIssue(
                severity="info",
                field="baseline_normal_operation",
                message="baseline_normal_operation section not found. "
                        "This section helps define normal operating ranges for sensors."
            ))
            return issues
        
        baseline = profile_data.get('baseline_normal_operation', {})
        
        # Check for sensor categories
        expected_categories = ['temperature', 'vibration', 'pressure', 'electrical']
        found_categories = [cat for cat in expected_categories if cat in baseline]
        
        if not found_categories:
            issues.append(ValidationIssue(
                severity="info",
                field="baseline_normal_operation",
                message="No standard sensor categories found (temperature, vibration, pressure, electrical). "
                        "Consider organizing sensors by category for better structure."
            ))
        
        return issues
    
    def _validate_baseline_operation(self, profile_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate baseline_normal_operation completeness"""
        issues = []
        
        baseline = profile_data.get('baseline_normal_operation', {})
        
        if not baseline:
            return issues  # Already reported in sensor_structure
        
        # Count sensors with alarm/trip thresholds
        sensors_with_alarms = 0
        total_sensors = 0
        
        for category, sensors_dict in baseline.items():
            if isinstance(sensors_dict, dict) and not category.startswith('_'):
                for sensor_name, sensor_config in sensors_dict.items():
                    if isinstance(sensor_config, dict) and not sensor_name.startswith('_'):
                        total_sensors += 1
                        if 'alarm' in sensor_config or 'trip' in sensor_config:
                            sensors_with_alarms += 1
        
        if total_sensors > 0 and sensors_with_alarms == 0:
            issues.append(ValidationIssue(
                severity="info",
                field="baseline_normal_operation",
                message="No alarm/trip thresholds defined for sensors. "
                        "Consider adding alarm thresholds for critical sensors."
            ))
        
        return issues
    
    def _validate_specifications(self, profile_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate specifications section"""
        issues = []
        
        if 'specifications' not in profile_data:
            issues.append(ValidationIssue(
                severity="info",
                field="specifications",
                message="specifications section not found. "
                        "Consider adding technical specifications for better documentation."
            ))
        
        return issues
    
    def get_validation_summary(self, issues: List[ValidationIssue]) -> Dict[str, Any]:
        """Generate validation summary with counts and severity breakdown"""
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        info_count = sum(1 for issue in issues if issue.severity == "info")
        
        return {
            "total_issues": len(issues),
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count,
            "can_proceed": error_count == 0,
            "issues": [issue.to_dict() for issue in issues]
        }
