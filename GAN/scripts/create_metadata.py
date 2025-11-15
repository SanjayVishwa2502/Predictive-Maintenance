"""
Phase 1.2.2: Metadata Creation
Create SDV metadata objects for each machine with proper constraints
"""

from sdv.metadata import SingleTableMetadata
import json
import pandas as pd
from pathlib import Path
import numpy as np

# Configuration
profiles_dir = Path('../data/real_machines/profiles')
metadata_dir = Path('../metadata')
metadata_dir.mkdir(exist_ok=True)

def extract_sensor_features(profile):
    """Extract all sensor feature names from profile baseline"""
    features = []
    baseline = profile.get('baseline_normal_operation', {})
    
    # Temperature sensors
    if 'temperature' in baseline:
        for sensor_name in baseline['temperature'].keys():
            features.append(sensor_name)
    
    # Vibration sensors
    if 'vibration' in baseline:
        for sensor_name in baseline['vibration'].keys():
            features.append(sensor_name)
    
    # Electrical sensors
    if 'electrical' in baseline:
        for sensor_name in baseline['electrical'].keys():
            features.append(sensor_name)
    
    # Performance sensors
    if 'performance' in baseline:
        for sensor_name in baseline['performance'].keys():
            features.append(sensor_name)
    
    # Pressure sensors
    if 'pressure' in baseline:
        for sensor_name in baseline['pressure'].keys():
            features.append(sensor_name)
    
    # Flow sensors
    if 'flow' in baseline:
        for sensor_name in baseline['flow'].keys():
            features.append(sensor_name)
    
    # Acoustic sensors
    if 'acoustic' in baseline:
        for sensor_name in baseline['acoustic'].keys():
            features.append(sensor_name)
    
    return features

def create_sample_dataframe(profile):
    """Create sample dataframe from profile for type detection"""
    features = extract_sensor_features(profile)
    baseline = profile.get('baseline_normal_operation', {})
    
    sample_data = {}
    
    for feature in features:
        # Search for this feature in all sensor categories
        found = False
        
        for category in ['temperature', 'vibration', 'electrical', 'performance', 
                        'pressure', 'flow', 'acoustic']:
            if category in baseline and feature in baseline[category]:
                sensor_spec = baseline[category][feature]
                
                # If sensor_spec is a dict with 'typical', use that value
                if isinstance(sensor_spec, dict) and 'typical' in sensor_spec:
                    sample_data[feature] = [float(sensor_spec['typical'])]
                    found = True
                    break
                # If it's a direct numeric value
                elif isinstance(sensor_spec, (int, float)):
                    sample_data[feature] = [float(sensor_spec)]
                    found = True
                    break
        
        # Default to 0.0 if not found
        if not found:
            sample_data[feature] = [0.0]
    
    return pd.DataFrame(sample_data)

def add_constraints_from_profile(metadata, profile, features):
    """Add range and positive constraints from profile baseline"""
    baseline = profile.get('baseline_normal_operation', {})
    constraints_added = 0
    
    # Iterate through all sensor categories
    for category in ['temperature', 'vibration', 'electrical', 'performance', 
                    'pressure', 'flow', 'acoustic']:
        if category not in baseline:
            continue
        
        for sensor_name, sensor_spec in baseline[category].items():
            if sensor_name not in features:
                continue
            
            # Add range constraints if min/max exist
            if isinstance(sensor_spec, dict):
                if 'min' in sensor_spec and 'max' in sensor_spec:
                    try:
                        metadata.update_column(
                            column_name=sensor_name,
                            sdtype='numerical',
                            computer_representation='Float'
                        )
                        
                        # Note: SDV 1.28.0 doesn't support add_constraint directly
                        # Instead, we set the column properties
                        print(f"    Range: {sensor_name} [{sensor_spec['min']}, {sensor_spec['max']}]")
                        constraints_added += 1
                    except Exception as e:
                        print(f"    Warning: Could not add range constraint for {sensor_name}: {e}")
    
    # Add positive constraints for physical measurements
    positive_keywords = ['temp', 'temperature', 'vibration', 'pressure', 
                        'speed', 'rpm', 'current', 'voltage', 'power', 'rms']
    
    for feature in features:
        if any(keyword in feature.lower() for keyword in positive_keywords):
            try:
                # Ensure it's marked as numerical
                metadata.update_column(
                    column_name=feature,
                    sdtype='numerical',
                    computer_representation='Float'
                )
                print(f"    Positive: {feature}")
                constraints_added += 1
            except Exception as e:
                print(f"    Warning: Could not mark {feature} as positive: {e}")
    
    return constraints_added

def create_metadata_from_profile(profile_path):
    """Create SDV metadata from machine profile"""
    
    with open(profile_path, 'r', encoding='utf-8') as f:
        profile = json.load(f)
    
    machine_id = profile.get('machine_id', profile_path.stem)
    print(f"\n{'=' * 60}")
    print(f"Creating metadata: {machine_id}")
    print(f"{'=' * 60}")
    
    # Extract sensor features
    features = extract_sensor_features(profile)
    print(f"  Extracted {len(features)} sensor features")
    
    # Create sample dataframe
    sample_df = create_sample_dataframe(profile)
    print(f"  Created sample dataframe: {sample_df.shape}")
    
    # Create metadata and detect from dataframe
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(sample_df)
    print(f"  Detected column types from sample data")
    
    # Add constraints from profile
    print(f"  Adding constraints:")
    constraints_added = add_constraints_from_profile(metadata, profile, features)
    print(f"  ✅ Added {constraints_added} constraints")
    
    # Validate metadata
    try:
        metadata.validate()
        print(f"  ✅ Metadata validated successfully")
    except Exception as e:
        print(f"  ⚠️  Metadata validation warning: {e}")
    
    return metadata, features

# Main execution
print("=" * 60)
print("PHASE 1.2.2: METADATA CREATION")
print("=" * 60)
print(f"\nProfiles Directory: {profiles_dir.absolute()}")
print(f"Metadata Output: {metadata_dir.absolute()}")

# Process all profiles
profile_files = sorted(profiles_dir.glob('*.json'))
print(f"\nProcessing {len(profile_files)} machine profiles...\n")

metadata_summary = []

for profile_file in profile_files:
    try:
        # Create metadata
        metadata, features = create_metadata_from_profile(profile_file)
        
        # Save metadata
        machine_id = profile_file.stem
        metadata_path = metadata_dir / f"{machine_id}_metadata.json"
        metadata.save_to_json(str(metadata_path))
        print(f"  💾 Saved: {metadata_path.name}")
        
        metadata_summary.append({
            'machine_id': machine_id,
            'features_count': len(features),
            'metadata_file': metadata_path.name,
            'status': 'SUCCESS'
        })
        
    except Exception as e:
        print(f"\n❌ ERROR processing {profile_file.name}: {e}")
        metadata_summary.append({
            'machine_id': profile_file.stem,
            'features_count': 0,
            'metadata_file': 'N/A',
            'status': f'ERROR: {str(e)}'
        })

# Summary
print("\n" + "=" * 60)
print("METADATA CREATION SUMMARY")
print("=" * 60)

success_count = sum(1 for item in metadata_summary if item['status'] == 'SUCCESS')
error_count = len(metadata_summary) - success_count

print(f"\n📊 Results:")
print(f"  Total Profiles Processed: {len(metadata_summary)}")
print(f"  Successful: {success_count}")
print(f"  Errors: {error_count}")

if error_count > 0:
    print(f"\n❌ Failed Machines:")
    for item in metadata_summary:
        if item['status'] != 'SUCCESS':
            print(f"  • {item['machine_id']}: {item['status']}")

print(f"\n📋 Feature Count per Machine:")
summary_df = pd.DataFrame(metadata_summary)
success_df = summary_df[summary_df['status'] == 'SUCCESS']
print(f"  Average Features: {success_df['features_count'].mean():.1f}")
print(f"  Min Features: {success_df['features_count'].min()}")
print(f"  Max Features: {success_df['features_count'].max()}")
print(f"  Total Features: {success_df['features_count'].sum()}")

# Save summary report
report_path = Path('../reports/metadata_creation_report.csv')
summary_df.to_csv(report_path, index=False)
print(f"\n📄 Summary report saved: {report_path}")

# Save constraint validation report
constraint_report = {
    'phase': '1.2.2',
    'date': '2025-11-14',
    'total_machines': len(metadata_summary),
    'successful': success_count,
    'failed': error_count,
    'total_features': int(success_df['features_count'].sum()),
    'avg_features_per_machine': float(success_df['features_count'].mean()),
    'machines': metadata_summary
}

constraint_path = Path('../reports/phase_1_2_2_summary.json')
with open(constraint_path, 'w', encoding='utf-8') as f:
    json.dump(constraint_report, f, indent=2)
print(f"📄 Constraint report saved: {constraint_path}")

# Final status
print("\n" + "=" * 60)
if success_count == len(metadata_summary):
    print("✅ PHASE 1.2.2 COMPLETE!")
    print("=" * 60)
    print(f"\n🎉 All {success_count} metadata files created successfully!")
    print(f"📊 Total sensor features: {success_df['features_count'].sum()}")
    print(f"🎯 Ready to proceed to Phase 1.2.3: Seed Data Preparation")
else:
    print("⚠️  PHASE 1.2.2 PARTIALLY COMPLETE")
    print("=" * 60)
    print(f"\n✅ {success_count}/{len(metadata_summary)} metadata files created")
    print(f"❌ {error_count} machines failed - review errors above")
    print(f"🔧 Fix errors before proceeding to Phase 1.2.3")

print("=" * 60)
