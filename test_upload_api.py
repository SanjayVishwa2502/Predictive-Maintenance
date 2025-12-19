"""
Test uploading duplicate machine via API
"""
import requests
import json

backend_url = "http://localhost:8000"

# Test 1: Try to upload existing machine (should FAIL)
print("=" * 70)
print("TEST 1: Upload cnc_haas_vf3_001 (DUPLICATE - Should FAIL)")
print("=" * 70)

duplicate_profile = {
    "machine_id": "cnc_haas_vf3_001",
    "machine_type": "cnc",
    "manufacturer": "Haas",
    "model": "VF-3",
    "sensors": [
        {
            "name": "spindle_temp",
            "display_name": "Spindle Temperature",
            "unit": "°C",
            "min_value": 20.0,
            "max_value": 80.0,
            "sensor_type": "temperature",
            "is_critical": True
        }
    ],
    "degradation_states": 4,
    "rul_min": 0,
    "rul_max": 1000
}

try:
    response = requests.post(
        f"{backend_url}/api/gan/profiles/upload",
        json=duplicate_profile
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 400:
        print("\n✅ SUCCESS: Duplicate machine rejected at upload stage!")
    else:
        print("\n❌ FAILURE: Duplicate machine was accepted!")
        
except Exception as e:
    print(f"Error: {e}")

print("\n")

# Test 2: Try to upload NEW machine (should PASS)
print("=" * 70)
print("TEST 2: Upload cnc_haas_vf4_002 (NEW - Should PASS)")
print("=" * 70)

new_profile = {
    "machine_id": "cnc_haas_vf4_002",
    "machine_type": "cnc",
    "manufacturer": "Haas",
    "model": "VF-4",
    "sensors": [
        {
            "name": "spindle_temp",
            "display_name": "Spindle Temperature",
            "unit": "°C",
            "min_value": 20.0,
            "max_value": 80.0,
            "sensor_type": "temperature",
            "is_critical": True
        }
    ],
    "degradation_states": 4,
    "rul_min": 0,
    "rul_max": 1000
}

try:
    response = requests.post(
        f"{backend_url}/api/gan/profiles/upload",
        json=new_profile
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS: New machine accepted!")
        # Clean up temp file
        profile_id = response.json().get('profile_id')
        print(f"Profile ID: {profile_id}")
    else:
        print("\n❌ FAILURE: New machine was rejected!")
        
except Exception as e:
    print(f"Error: {e}")
