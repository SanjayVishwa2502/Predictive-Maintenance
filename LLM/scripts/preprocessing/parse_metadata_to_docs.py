"""
Convert 27 machine JSON metadata files to searchable text documents
Phase 3.1.1: Parse Machine Metadata

Note: TVAE metadata contains sensor schemas, not machine specs.
We'll generate structured machine documentation from filenames and sensor schemas.
"""
import json
from pathlib import Path

# Machine knowledge base templates
MACHINE_TEMPLATES = {
    'motor': {
        'type': 'Electric Motor',
        'common_failures': ['Bearing wear', 'Winding insulation failure', 'Overheating', 'Shaft misalignment'],
        'maintenance': ['Lubricate bearings every 3 months', 'Inspect windings annually', 'Check vibration monthly', 'Clean cooling vents quarterly']
    },
    'pump': {
        'type': 'Centrifugal Pump',
        'common_failures': ['Cavitation', 'Seal leakage', 'Impeller wear', 'Bearing failure'],
        'maintenance': ['Replace seals every 6 months', 'Check alignment quarterly', 'Monitor vibration weekly', 'Inspect impeller annually']
    },
    'compressor': {
        'type': 'Air Compressor',
        'common_failures': ['Oil contamination', 'Valve failure', 'Bearing wear', 'Overheating'],
        'maintenance': ['Change oil every 2000 hours', 'Replace filters monthly', 'Check pressure relief valve quarterly', 'Inspect bearings every 6 months']
    },
    'cnc': {
        'type': 'CNC Machine',
        'common_failures': ['Spindle bearing wear', 'Tool changer malfunction', 'Coolant pump failure', 'Servo motor issues'],
        'maintenance': ['Check tool alignment weekly', 'Lubricate ways daily', 'Inspect spindle bearings monthly', 'Clean coolant system weekly']
    },
    'hydraulic': {
        'type': 'Hydraulic System',
        'common_failures': ['Fluid leakage', 'Pump wear', 'Valve malfunction', 'Contamination'],
        'maintenance': ['Change hydraulic fluid annually', 'Replace filters quarterly', 'Check hoses monthly', 'Inspect seals every 6 months']
    },
    'conveyor': {
        'type': 'Conveyor Belt System',
        'common_failures': ['Belt wear', 'Motor failure', 'Bearing seizure', 'Misalignment'],
        'maintenance': ['Check belt tension weekly', 'Lubricate bearings monthly', 'Inspect rollers quarterly', 'Clean system monthly']
    },
    'fan': {
        'type': 'Industrial Fan',
        'common_failures': ['Blade wear', 'Bearing failure', 'Motor overheating', 'Vibration issues'],
        'maintenance': ['Balance blades quarterly', 'Lubricate bearings monthly', 'Check motor temperature weekly', 'Clean blades monthly']
    },
    'cooling_tower': {
        'type': 'Cooling Tower',
        'common_failures': ['Scale buildup', 'Fan motor failure', 'Pump malfunction', 'Water contamination'],
        'maintenance': ['Treat water monthly', 'Clean basin quarterly', 'Inspect fan annually', 'Check pump monthly']
    },
    'robot': {
        'type': 'Industrial Robot',
        'common_failures': ['Joint motor failure', 'Encoder malfunction', 'Cable wear', 'Gripper issues'],
        'maintenance': ['Check encoder signals monthly', 'Lubricate joints quarterly', 'Inspect cables weekly', 'Calibrate annually']
    },
    'transformer': {
        'type': 'Power Transformer',
        'common_failures': ['Insulation breakdown', 'Overheating', 'Oil contamination', 'Winding failure'],
        'maintenance': ['Test oil annually', 'Monitor temperature daily', 'Inspect bushings quarterly', 'Check grounding monthly']
    },
    'turbofan': {
        'type': 'Turbofan Engine',
        'common_failures': ['Blade erosion', 'Bearing wear', 'Combustion issues', 'Vibration'],
        'maintenance': ['Inspect blades after each flight', 'Monitor vibration continuously', 'Check bearings every 500 hours', 'Clean compressor quarterly']
    }
}

def generate_machine_doc(machine_id, tvae_metadata_path):
    """Generate machine documentation from machine ID and TVAE metadata"""
    
    # Extract machine type from ID
    machine_type = None
    for key in MACHINE_TEMPLATES.keys():
        if key in machine_id:
            machine_type = key
            break
    
    if not machine_type:
        machine_type = 'motor'  # default
    
    template = MACHINE_TEMPLATES[machine_type]
    
    # Load TVAE metadata to get sensor list
    with open(tvae_metadata_path) as f:
        tvae_meta = json.load(f)
    
    sensors = list(tvae_meta['columns'].keys())
    
    # Parse manufacturer and model from ID
    parts = machine_id.replace('_metadata', '').split('_')
    manufacturer = parts[1] if len(parts) > 1 else 'Unknown'
    model = parts[2] if len(parts) > 2 else 'Standard'
    
    doc = f"""
Machine ID: {machine_id.replace('_metadata', '')}
Type: {template['type']}
Manufacturer: {manufacturer.upper()}
Model: {model.upper()}

SPECIFICATIONS:
- Sensors Monitored: {len(sensors)}
- Data Collection: Real-time, 1-hour intervals
- Operational Status: Active monitoring

SENSORS ({len(sensors)}):
{chr(10).join(f"- {s}" for s in sensors[:15])}  # Show first 15 sensors
{f"... and {len(sensors) - 15} more sensors" if len(sensors) > 15 else ""}

COMMON FAILURE MODES:
{chr(10).join(f"- {fm}" for fm in template['common_failures'])}

MAINTENANCE PROCEDURES:
{chr(10).join(f"- {m}" for m in template['maintenance'])}

SAFETY NOTES:
- Always follow lockout/tagout procedures
- Check for abnormal vibrations before operation
- Monitor temperature readings regularly
- Report unusual noises or performance degradation immediately
"""
    return doc

def batch_parse_metadata():
    """Parse all 27 machines"""
    # Use absolute paths from project root
    project_root = Path(__file__).resolve().parents[3]
    metadata_dir = project_root / "GAN" / "metadata"
    output_dir = project_root / "LLM" / "data" / "knowledge_base" / "machines"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parsed_count = 0
    
    for json_file in metadata_dir.glob("*.json"):
        doc = generate_machine_doc(json_file.stem, json_file)
        output_file = output_dir / f"{json_file.stem.replace('_metadata', '')}.txt"
        output_file.write_text(doc, encoding='utf-8')
        print(f"✓ {json_file.stem.replace('_metadata', '')}")
        parsed_count += 1
    
    print(f"\n{'='*60}")
    print(f"Phase 3.1.1 Complete: Parsed {parsed_count} machine metadata files")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Phase 3.1.1: Parsing Machine Metadata to Text Documents")
    print("="*60)
    batch_parse_metadata()
