/**
 * Profile Validation Rules - Phase 3.7.6.2 Enhancement
 * 
 * Implements 14 intelligent parsing and validation rules for machine profiles
 * Based on the template parsing fallback rules
 */

import type { MachineProfile, ValidationError } from '../types/gan.types';

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
  normalized: Partial<MachineProfile> | null;
}

/**
 * Comprehensive validation with 14 fallback rules
 */
export function validateMachineProfile(data: any): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];
  let normalized = { ...data };

  // RULE 1: If 'machine_id' is missing → Generate from: <category>_<manufacturer>_<model>_001
  if (!data.machine_id || data.machine_id.includes('*** REQUIRED')) {
    if (data.category && data.manufacturer && data.model) {
      const cleanCategory = data.category.toLowerCase().replace(/\s+/g, '_');
      const cleanManufacturer = data.manufacturer.toLowerCase().replace(/\s+/g, '_');
      const cleanModel = data.model.toLowerCase().replace(/\s+/g, '_').substring(0, 10);
      normalized.machine_id = `${cleanCategory}_${cleanManufacturer}_${cleanModel}_001`;
      warnings.push({
        field: 'machine_id',
        message: `Machine ID auto-generated: ${normalized.machine_id}`,
      });
    } else {
      errors.push({
        field: 'machine_id',
        message: 'Machine ID is required. Format: <type>_<manufacturer>_<model>_<id>',
      });
    }
  }

  // RULE 2: If 'category' is missing → Try to infer from machine_id prefix
  if (!data.category || data.category.includes('*** REQUIRED')) {
    if (data.machine_id && !data.machine_id.includes('*** REQUIRED')) {
      const prefix = data.machine_id.split('_')[0];
      normalized.category = prefix;
      warnings.push({
        field: 'category',
        message: `Category inferred from machine_id: ${prefix}`,
      });
    } else if (data.machine_type) {
      normalized.category = data.machine_type;
      warnings.push({
        field: 'category',
        message: `Category set from machine_type field`,
      });
    } else {
      errors.push({
        field: 'category',
        message: 'Category is required (e.g., motor, pump, cnc, compressor)',
      });
    }
  }

  // RULE 3: If 'baseline_normal_operation' is missing → Look for alternatives
  if (!data.baseline_normal_operation) {
    if (data.operating_parameters) {
      normalized.baseline_normal_operation = data.operating_parameters;
      warnings.push({
        field: 'baseline_normal_operation',
        message: 'Using operating_parameters as baseline_normal_operation',
      });
    } else if (data.sensor_data) {
      normalized.baseline_normal_operation = data.sensor_data;
      warnings.push({
        field: 'baseline_normal_operation',
        message: 'Using sensor_data as baseline_normal_operation',
      });
    } else if (data.normal_conditions) {
      normalized.baseline_normal_operation = data.normal_conditions;
      warnings.push({
        field: 'baseline_normal_operation',
        message: 'Using normal_conditions as baseline_normal_operation',
      });
    } else {
      errors.push({
        field: 'baseline_normal_operation',
        message: 'Baseline normal operation data is required for GAN training',
      });
    }
  }

  // RULE 4-5: Validate sensor data structure and auto-generate missing min/max
  if (normalized.baseline_normal_operation) {
    normalized.baseline_normal_operation = normalizeSensorData(
      normalized.baseline_normal_operation,
      warnings
    );
  }

  // RULE 6-8: Auto-group flat sensor data into categories
  if (normalized.baseline_normal_operation) {
    normalized.baseline_normal_operation = autoGroupSensors(
      normalized.baseline_normal_operation,
      warnings
    );
  }

  // RULE 9: Infer units from field names if missing
  if (normalized.baseline_normal_operation) {
    inferUnits(normalized.baseline_normal_operation);
  }

  // RULE 10: Validate manufacturer has correct formatting (spaces preserved)
  if (!data.manufacturer || data.manufacturer.includes('*** REQUIRED')) {
    errors.push({
      field: 'manufacturer',
      message: 'Manufacturer name is required (e.g., Siemens, DMG MORI, Atlas Copco)',
    });
  }

  // Validate model field
  if (!data.model || data.model.includes('*** REQUIRED')) {
    errors.push({
      field: 'model',
      message: 'Model number is required',
    });
  }

  // RULE 11: Accept alternative field names
  if (data.name && !normalized.machine_id) {
    normalized.machine_id = data.name;
  }
  if (data.specs && !normalized.specifications) {
    normalized.specifications = data.specs;
  }

  // RULE 12: Check for flat structure and auto-group
  if (isFlatStructure(data)) {
    const grouped = groupFlatSensors(data);
    if (Object.keys(grouped).length > 0) {
      normalized.baseline_normal_operation = grouped;
      warnings.push({
        field: 'baseline_normal_operation',
        message: 'Sensors were auto-grouped from flat structure',
      });
    }
  }

  // RULE 13: fault_signatures is optional
  if (!data.fault_signatures) {
    warnings.push({
      field: 'fault_signatures',
      message: 'Fault signatures not provided (optional but recommended)',
    });
  }

  // RULE 14: Detect and normalize multiple JSON formats
  const formatDetection = detectFormat(data);
  if (formatDetection.needsNormalization) {
    warnings.push({
      field: '_format',
      message: `Detected ${formatDetection.format} format, normalized to standard format`,
    });
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    normalized: errors.length === 0 ? (normalized as Partial<MachineProfile>) : null,
  };
}

/**
 * Normalize sensor data: auto-generate min/max from typical values
 */
function normalizeSensorData(sensorData: any, warnings: ValidationError[]): any {
  const normalized = { ...sensorData };

  const processSensorGroup = (group: any, groupName: string) => {
    if (!group || typeof group !== 'object') return group;

    const processedGroup = { ...group };
    for (const [key, value] of Object.entries(processedGroup)) {
      if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
        // Check if this is a sensor definition
        if ('typical' in value && (!('min' in value) || !('max' in value))) {
          const typical = Number(value.typical);
          if (!isNaN(typical)) {
            if (!('min' in value)) {
              processedGroup[key].min = typical * 0.8; // 20% below typical
            }
            if (!('max' in value)) {
              processedGroup[key].max = typical * 1.2; // 20% above typical
            }
            warnings.push({
              field: `${groupName}.${key}`,
              message: 'Auto-generated min/max from typical value (±20%)',
            });
          }
        }
        // Single value sensor (just a number)
        else if (typeof value === 'number') {
          processedGroup[key] = {
            typical: value,
            min: value * 0.8,
            max: value * 1.2,
            unit: inferUnitFromFieldName(key),
          };
          warnings.push({
            field: `${groupName}.${key}`,
            message: 'Converted single value to sensor object with min/typical/max',
          });
        }
      }
    }
    return processedGroup;
  };

  // Process each sensor category
  for (const [category, group] of Object.entries(normalized)) {
    if (typeof group === 'object' && group !== null && !Array.isArray(group)) {
      normalized[category] = processSensorGroup(group, category);
    }
  }

  return normalized;
}

/**
 * Auto-group sensors by category if not already grouped
 */
function autoGroupSensors(sensorData: any, warnings: ValidationError[]): any {
  const categories = ['temperature', 'vibration', 'electrical', 'mechanical', 'pressure', 'hydraulic', 'acoustic'];
  
  // Check if already grouped
  const hasCategories = categories.some(cat => cat in sensorData);
  if (hasCategories) {
    return sensorData; // Already grouped
  }

  // Auto-group
  const grouped: any = {};
  let hasGrouped = false;

  for (const [key, value] of Object.entries(sensorData)) {
    if (key.startsWith('_')) continue; // Skip comments

    const lowerKey = key.toLowerCase();
    
    // Determine category
    let category = 'other';
    if (lowerKey.includes('temp') || lowerKey.includes('temperature') || lowerKey.endsWith('_t') || lowerKey.endsWith('_c')) {
      category = 'temperature';
    } else if (lowerKey.includes('vib') || lowerKey.includes('vibration') || lowerKey.includes('rms')) {
      category = 'vibration';
    } else if (lowerKey.includes('current') || lowerKey.includes('voltage') || lowerKey.includes('power') || lowerKey.includes('_a') || lowerKey.includes('_v') || lowerKey.includes('_kw')) {
      category = 'electrical';
    } else if (lowerKey.includes('speed') || lowerKey.includes('rpm') || lowerKey.includes('torque')) {
      category = 'mechanical';
    } else if (lowerKey.includes('pressure') || lowerKey.includes('_bar') || lowerKey.includes('_psi')) {
      category = 'pressure';
    } else if (lowerKey.includes('flow') || lowerKey.includes('rate')) {
      category = 'hydraulic';
    } else if (lowerKey.includes('sound') || lowerKey.includes('noise') || lowerKey.includes('db')) {
      category = 'acoustic';
    }

    if (!grouped[category]) {
      grouped[category] = {};
    }
    grouped[category][key] = value;
    hasGrouped = true;
  }

  if (hasGrouped) {
    warnings.push({
      field: 'sensors',
      message: `Auto-grouped ${Object.keys(grouped).length} sensor categories`,
    });
    return grouped;
  }

  return sensorData;
}

/**
 * Infer units from field names
 */
function inferUnits(sensorData: any): void {
  const unitPatterns = [
    { pattern: /_(rpm|RPM)$/, unit: 'rpm' },
    { pattern: /_(kw|KW|kW)$/, unit: 'kW' },
    { pattern: /_(bar|BAR)$/, unit: 'bar' },
    { pattern: /_(c|C|celsius)$/i, unit: '°C' },
    { pattern: /_(mm_s|mms)$/i, unit: 'mm/s' },
    { pattern: /_(a|A|amp|amps)$/i, unit: 'A' },
    { pattern: /_(v|V|volt|volts)$/i, unit: 'V' },
    { pattern: /_(nm|Nm)$/i, unit: 'Nm' },
    { pattern: /_(m3_h|m3h)$/i, unit: 'm³/h' },
    { pattern: /_(db|dB|dBA)$/i, unit: 'dBA' },
  ];

  const processGroup = (group: any) => {
    for (const [key, value] of Object.entries(group)) {
      const sensor = value as any;
      if (typeof sensor === 'object' && sensor !== null && !sensor.unit) {
        for (const { pattern, unit } of unitPatterns) {
          if (pattern.test(key)) {
            sensor.unit = unit;
            break;
          }
        }
      }
    }
  };

  for (const group of Object.values(sensorData)) {
    if (typeof group === 'object' && group !== null) {
      processGroup(group);
    }
  }
}

function inferUnitFromFieldName(fieldName: string): string {
  const lowerField = fieldName.toLowerCase();
  if (lowerField.includes('temp') || lowerField.endsWith('_c')) return '°C';
  if (lowerField.includes('rpm')) return 'rpm';
  if (lowerField.includes('kw')) return 'kW';
  if (lowerField.includes('bar')) return 'bar';
  if (lowerField.includes('current') || lowerField.endsWith('_a')) return 'A';
  if (lowerField.includes('voltage') || lowerField.endsWith('_v')) return 'V';
  if (lowerField.includes('torque') || lowerField.includes('nm')) return 'Nm';
  if (lowerField.includes('flow')) return 'm³/h';
  if (lowerField.includes('vib')) return 'mm/s';
  return '';
}

/**
 * Check if structure is flat (all sensors at top level)
 */
function isFlatStructure(data: any): boolean {
  // Look for sensor-like fields at top level
  const sensorPatterns = ['temp', 'vibration', 'current', 'voltage', 'pressure', 'speed', 'torque', 'flow'];
  let sensorCount = 0;

  for (const key of Object.keys(data)) {
    if (key.startsWith('_')) continue;
    const lowerKey = key.toLowerCase();
    if (sensorPatterns.some(pattern => lowerKey.includes(pattern))) {
      sensorCount++;
    }
  }

  return sensorCount >= 3; // If 3+ sensor-like fields at top level, consider it flat
}

/**
 * Group flat sensor data into categories
 */
function groupFlatSensors(data: any): any {
  const grouped: any = {};

  for (const [key, value] of Object.entries(data)) {
    if (key.startsWith('_') || ['machine_id', 'manufacturer', 'model', 'category', 'specifications', 'notes'].includes(key)) {
      continue;
    }

    const lowerKey = key.toLowerCase();
    let category = 'other';

    if (lowerKey.includes('temp') || lowerKey.endsWith('_c')) {
      category = 'temperature';
    } else if (lowerKey.includes('vib')) {
      category = 'vibration';
    } else if (lowerKey.includes('current') || lowerKey.includes('voltage') || lowerKey.includes('power')) {
      category = 'electrical';
    } else if (lowerKey.includes('speed') || lowerKey.includes('torque')) {
      category = 'mechanical';
    } else if (lowerKey.includes('pressure')) {
      category = 'pressure';
    }

    if (!grouped[category]) {
      grouped[category] = {};
    }
    grouped[category][key] = value;
  }

  return grouped;
}

/**
 * Detect profile format
 */
function detectFormat(data: any): { format: string; needsNormalization: boolean } {
  if (data._parsing_fallback_rules) {
    return { format: 'Full Template Format', needsNormalization: false };
  }
  if (data.metadata && data.tables) {
    return { format: 'SDV Metadata Format', needsNormalization: true };
  }
  if (!data.baseline_normal_operation && isFlatStructure(data)) {
    return { format: 'Flat JSON Format', needsNormalization: true };
  }
  if (data.machine_id && data.manufacturer && data.model) {
    return { format: 'Simplified Format', needsNormalization: false };
  }
  return { format: 'Custom Format', needsNormalization: true };
}
