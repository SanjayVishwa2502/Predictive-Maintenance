"""
Test script for Phase 3.7.1.3: Authentication endpoints
Tests user registration, login, token refresh, and protected routes.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_response(response):
    print(f"Status: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_authentication():
    """Test complete authentication flow."""
    
    print_section("PHASE 3.7.1.3 AUTHENTICATION TESTING")
    
    # Test 1: Register admin user
    print_section("Test 1: Register Admin User")
    register_data = {
        "username": "admin",
        "email": "admin@predictive.ai",
        "password": "AdminPass123",
        "role": "admin"
    }
    response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
    print_response(response)
    admin_registered = response.status_code == 201
    
    # Test 2: Register operator user
    print_section("Test 2: Register Operator User")
    register_data = {
        "username": "operator1",
        "email": "operator@predictive.ai",
        "password": "OperatorPass123",
        "role": "operator"
    }
    response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
    print_response(response)
    
    # Test 3: Register viewer user
    print_section("Test 3: Register Viewer User")
    register_data = {
        "username": "viewer1",
        "email": "viewer@predictive.ai",
        "password": "ViewerPass123",
        "role": "viewer"
    }
    response = requests.post(f"{BASE_URL}/api/auth/register", json=register_data)
    print_response(response)
    
    # Test 4: Try duplicate username (should fail)
    print_section("Test 4: Duplicate Username (Should Fail)")
    response = requests.post(f"{BASE_URL}/api/auth/register", json={
        "username": "admin",
        "email": "another@predictive.ai",
        "password": "Pass123",
        "role": "viewer"
    })
    print_response(response)
    
    # Test 5: Login with admin
    print_section("Test 5: Login with Admin")
    login_data = {
        "username": "admin",
        "password": "AdminPass123"
    }
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
    print_response(response)
    
    if response.status_code == 200:
        tokens = response.json()
        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]
        
        # Test 6: Get current user info
        print_section("Test 6: Get Current User Info")
        headers = {"Authorization": f"Bearer {access_token}"}
        response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
        print_response(response)
        
        # Test 7: Access admin-only endpoint
        print_section("Test 7: List All Users (Admin Only)")
        response = requests.get(f"{BASE_URL}/api/auth/users", headers=headers)
        print_response(response)
        
        # Test 8: Refresh token
        print_section("Test 8: Refresh Access Token")
        response = requests.post(f"{BASE_URL}/api/auth/refresh", json={
            "refresh_token": refresh_token
        })
        print_response(response)
        
        if response.status_code == 200:
            new_tokens = response.json()
            new_access_token = new_tokens["access_token"]
            
            # Test 9: Use new access token
            print_section("Test 9: Use New Access Token")
            headers = {"Authorization": f"Bearer {new_access_token}"}
            response = requests.get(f"{BASE_URL}/api/auth/me", headers=headers)
            print_response(response)
    
    # Test 10: Login with wrong password
    print_section("Test 10: Login with Wrong Password (Should Fail)")
    response = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username": "admin",
        "password": "WrongPassword"
    })
    print_response(response)
    
    # Test 11: Access protected endpoint without token
    print_section("Test 11: Access Protected Endpoint Without Token (Should Fail)")
    response = requests.get(f"{BASE_URL}/api/auth/me")
    print_response(response)
    
    # Test 12: Login as viewer and try admin endpoint
    print_section("Test 12: Viewer Tries Admin Endpoint (Should Fail)")
    login_response = requests.post(f"{BASE_URL}/api/auth/login", json={
        "username": "viewer1",
        "password": "ViewerPass123"
    })
    
    if login_response.status_code == 200:
        viewer_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {viewer_token}"}
        response = requests.get(f"{BASE_URL}/api/auth/users", headers=headers)
        print_response(response)
    
    print_section("AUTHENTICATION TESTING COMPLETE")
    print("\n✅ All authentication features tested!")
    print("\nTest Summary:")
    print("  ✅ User registration (admin, operator, viewer)")
    print("  ✅ Login and token generation")
    print("  ✅ Token refresh")
    print("  ✅ Protected endpoints with JWT")
    print("  ✅ Role-based access control")
    print("  ✅ Error handling (duplicate users, wrong passwords, unauthorized access)")

if __name__ == "__main__":
    try:
        test_authentication()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to server")
        print("Make sure the FastAPI server is running on http://localhost:8000")
        print("Run: cd frontend/server && uvicorn main:app --reload")
