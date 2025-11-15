"""
Phase 1.2.1: Machine Profile Audit
Review and validate all 21 machine profiles for completeness
"""

import json
from pathlib import Path
import pandas as pd

# Configuration
profiles_dir = Path('../data/real_machines/profiles')

# Define required fields based on actual profile structure
required_fields = {
    'machine_id': 'Unique machine identifier',
    'manufacturer': 'Equipment manufacturer',
    'model': 'Model designation',
    'category': 'Machine category',
    'specifications': 'Technical specifications',
    'baseline_normal_operation': 'Normal operating ranges'
}

optional_fields = {
    'application': 'Application description',
    'data_source': 'Data source reference'
}

print("=" * 60)
print("PHASE 1.2.1: MACHINE PROFILE AUDIT")
print("=" * 60)
print(f"\nProfiles Directory: {profiles_dir.absolute()}")
print(f"Expected Machines: 20+ profiles")

# Storage for results
complete_profiles = []
incomplete_profiles = []
profile_details = []

# Audit all profiles
profile_files = sorted(profiles_dir.glob('*.json'))
print(f"\nFound {len(profile_files)} profile files\n")

for profile_file in profile_files:
    print(f"\n{'=' * 60}")
    print(f"Auditing: {profile_file.name}")
    print(f"{'=' * 60}")
    
    try:
        with open(profile_file, 'r', encoding='utf-8') as f:
            profile = json.load(f)
        
        machine_id = profile.get('machine_id', profile_file.stem)
        
        # Check required fields
        missing_fields = []
        present_fields = []
        
        for field, description in required_fields.items():
            if field in profile and profile[field]:
                present_fields.append(field)
                print(f"  ✅ {field}: {description}")
            else:
                missing_fields.append(field)
                print(f"  ❌ {field}: {description} [MISSING]")
        
        # Check specifications detail
        if 'specifications' in profile:
            specs = profile['specifications']
            print(f"  📊 Specifications: {len(specs)} items")
            # Show key specs
            key_specs = ['power_rating_kW', 'rated_speed_rpm', 'voltage_V', 
                        'flow_rate_m3_h', 'pressure_bar', 'capacity_kW']
            found_specs = [s for s in key_specs if s in specs]
            if found_specs:
                print(f"     Found: {', '.join(found_specs)}")
        
        # Check baseline normal operation
        if 'baseline_normal_operation' in profile:
            baseline = profile['baseline_normal_operation']
            # Count sensor categories
            sensor_categories = []
            if 'temperature' in baseline:
                temp_sensors = len(baseline['temperature'])
                sensor_categories.append(f"Temperature: {temp_sensors}")
            if 'vibration' in baseline:
                vib_sensors = len(baseline['vibration'])
                sensor_categories.append(f"Vibration: {vib_sensors}")
            if 'electrical' in baseline:
                elec_sensors = len(baseline['electrical'])
                sensor_categories.append(f"Electrical: {elec_sensors}")
            if 'performance' in baseline:
                perf_sensors = len(baseline['performance'])
                sensor_categories.append(f"Performance: {perf_sensors}")
            if 'pressure' in baseline:
                press_sensors = len(baseline['pressure'])
                sensor_categories.append(f"Pressure: {press_sensors}")
            
            print(f"  📈 Baseline Sensors: {', '.join(sensor_categories)}")
            
            # Count total sensor features
            total_features = sum([
                len(baseline.get('temperature', {})),
                len(baseline.get('vibration', {})),
                len(baseline.get('electrical', {})),
                len(baseline.get('performance', {})),
                len(baseline.get('pressure', {})),
                len(baseline.get('flow', {})),
                len(baseline.get('acoustic', {}))
            ])
            print(f"  🎯 Total Features: {total_features}")
        else:
            total_features = 0
        
        # Check failure modes
        failure_modes = 0
        if 'failure_modes' in profile:
            failure_modes = len(profile['failure_modes'])
            print(f"  ⚠️  Failure Modes: {failure_modes}")
        
        # Determine completeness
        if missing_fields:
            status = "INCOMPLETE"
            incomplete_profiles.append({
                'machine_id': machine_id,
                'missing': missing_fields,
                'file': profile_file.name
            })
            print(f"\n  ❌ Status: {status}")
            print(f"     Missing: {', '.join(missing_fields)}")
        else:
            status = "COMPLETE"
            complete_profiles.append(machine_id)
            print(f"\n  ✅ Status: {status}")
        
        # Store details
        profile_details.append({
            'machine_id': machine_id,
            'file': profile_file.name,
            'category': profile.get('category', 'unknown'),
            'manufacturer': profile.get('manufacturer', 'unknown'),
            'model': profile.get('model', 'unknown'),
            'status': status,
            'total_features': total_features,
            'failure_modes': failure_modes,
            'has_specifications': 'specifications' in profile,
            'has_baseline': 'baseline_normal_operation' in profile,
            'missing_fields': ', '.join(missing_fields) if missing_fields else 'None'
        })
        
    except Exception as e:
        print(f"  ❌ ERROR reading profile: {e}")
        incomplete_profiles.append({
            'machine_id': profile_file.stem,
            'missing': ['FILE_READ_ERROR'],
            'file': profile_file.name
        })

# Summary Statistics
print("\n" + "=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)

print(f"\n📊 Profile Statistics:")
print(f"  Total Profiles Found: {len(profile_files)}")
print(f"  Complete Profiles: {len(complete_profiles)}")
print(f"  Incomplete Profiles: {len(incomplete_profiles)}")
print(f"  Completion Rate: {len(complete_profiles)/len(profile_files)*100:.1f}%")

# Category breakdown
df = pd.DataFrame(profile_details)
print(f"\n📋 Category Breakdown:")
category_counts = df['category'].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count} machines")

print(f"\n🎯 Feature Statistics:")
print(f"  Average Features per Machine: {df['total_features'].mean():.1f}")
print(f"  Min Features: {df['total_features'].min()}")
print(f"  Max Features: {df['total_features'].max()}")
print(f"  Total Features Across All Machines: {df['total_features'].sum()}")

# Incomplete profiles action items
if incomplete_profiles:
    print(f"\n⚠️  ACTION REQUIRED:")
    print(f"  {len(incomplete_profiles)} profiles need attention:\n")
    for item in incomplete_profiles:
        print(f"  ❌ {item['machine_id']}")
        print(f"     File: {item['file']}")
        print(f"     Missing: {', '.join(item['missing'])}\n")
else:
    print(f"\n✅ All profiles complete! Ready for Phase 1.2.2")

# Priority order for training
print("\n" + "=" * 60)
print("TRAINING PRIORITY ORDER")
print("=" * 60)

# Prioritize complete profiles with most features
priority_df = df[df['status'] == 'COMPLETE'].sort_values('total_features', ascending=False)

print("\n🎯 Recommended Training Order (Complete profiles, sorted by feature count):\n")
for idx, row in priority_df.iterrows():
    print(f"  {idx+1:2d}. {row['machine_id']:40s} ({row['total_features']:2d} features, {row['category']})")

# Save detailed audit report
report_path = Path('../reports/profile_audit_report.csv')
df.to_csv(report_path, index=False)
print(f"\n📄 Detailed audit report saved: {report_path}")

# Save priority order
priority_path = Path('../reports/training_priority_order.txt')
with open(priority_path, 'w', encoding='utf-8') as f:
    f.write("TRAINING PRIORITY ORDER\n")
    f.write("=" * 60 + "\n\n")
    f.write("Priority: Complete profiles sorted by feature count\n\n")
    for idx, row in priority_df.iterrows():
        f.write(f"{idx+1:2d}. {row['machine_id']} ({row['total_features']} features)\n")
    f.write(f"\nTotal machines ready for training: {len(priority_df)}\n")

print(f"📄 Priority order saved: {priority_path}")

# Save summary JSON
summary = {
    'audit_date': '2025-11-14',
    'phase': '1.2.1',
    'total_profiles': len(profile_files),
    'complete_profiles': len(complete_profiles),
    'incomplete_profiles': len(incomplete_profiles),
    'completion_rate': len(complete_profiles)/len(profile_files)*100,
    'total_features': int(df['total_features'].sum()),
    'avg_features_per_machine': float(df['total_features'].mean()),
    'categories': category_counts.to_dict(),
    'complete_profile_list': complete_profiles,
    'incomplete_profile_list': [item['machine_id'] for item in incomplete_profiles],
    'ready_for_training': len(complete_profiles) >= 20
}

summary_path = Path('../reports/phase_1_2_1_summary.json')
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f"📄 Summary JSON saved: {summary_path}")

# Final status
print("\n" + "=" * 60)
if len(complete_profiles) >= 20:
    print("✅ PHASE 1.2.1 COMPLETE!")
    print("=" * 60)
    print(f"\n🎉 {len(complete_profiles)} profiles validated and ready!")
    print(f"📊 Total features available: {df['total_features'].sum()}")
    print(f"🎯 Ready to proceed to Phase 1.2.2: Metadata Creation")
else:
    print("⚠️  PHASE 1.2.1 INCOMPLETE")
    print("=" * 60)
    print(f"\n❌ Only {len(complete_profiles)}/20 profiles complete")
    print(f"🔧 Please complete {20 - len(complete_profiles)} more profiles before proceeding")

print("=" * 60)
