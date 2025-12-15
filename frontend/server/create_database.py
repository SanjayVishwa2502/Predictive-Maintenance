"""
Create PostgreSQL database for predictive maintenance dashboard.
Phase 3.7.1.2: Database Setup
"""
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
import getpass

def create_database():
    """Create predictive_maintenance database if it doesn't exist."""
    
    # Get PostgreSQL credentials
    print("PostgreSQL Database Creation")
    print("-" * 50)
    postgres_user = input("Enter PostgreSQL username (default: postgres): ").strip() or "postgres"
    postgres_password = getpass.getpass("Enter PostgreSQL password: ")
    postgres_port = input("Enter PostgreSQL port (default: 5433): ").strip() or "5433"
    
    # Connect to default postgres database
    admin_url = f"postgresql://{postgres_user}:{postgres_password}@localhost:{postgres_port}/postgres"
    
    try:
        print(f"\nConnecting to PostgreSQL on port {postgres_port}...")
        engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
        
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(
                "SELECT 1 FROM pg_database WHERE datname='predictive_maintenance'"
            ))
            
            if result.fetchone():
                print("✅ Database 'predictive_maintenance' already exists")
            else:
                # Create database
                print("Creating database 'predictive_maintenance'...")
                conn.execute(text("CREATE DATABASE predictive_maintenance"))
                print("✅ Database 'predictive_maintenance' created successfully")
        
        # Update .env file with correct credentials
        env_path = "c:/Projects/Predictive Maintenance/frontend/server/.env"
        with open(env_path, 'r') as f:
            env_content = f.read()
        
        # Replace DATABASE_URL
        new_url = f"postgresql://{postgres_user}:{postgres_password}@localhost:{postgres_port}/predictive_maintenance"
        lines = env_content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('DATABASE_URL='):
                lines[i] = f'DATABASE_URL={new_url}'
                break
        
        with open(env_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ Updated .env file with DATABASE_URL")
        print(f"\nDatabase connection string:")
        print(f"postgresql://{postgres_user}:***@localhost:{postgres_port}/predictive_maintenance")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = create_database()
    sys.exit(0 if success else 1)
