"""
Verification script for Phase 3.7.1.2: Database Setup
Tests database connection and lists all created tables.
"""
import sys
from sqlalchemy import inspect, text

sys.path.insert(0, ".")
from database import engine, check_db_connection
from config import settings

def verify_database():
    """Verify database setup is complete."""
    print("=" * 70)
    print("PHASE 3.7.1.2 VERIFICATION: Database Setup")
    print("=" * 70)
    
    # Test connection
    print("\n1. Testing database connection...")
    if not check_db_connection():
        print("❌ Database connection failed!")
        return False
    
    # List all tables
    print("\n2. Verifying tables...")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    expected_tables = [
        "machines",
        "gan_training_jobs",
        "predictions",
        "explanations",
        "model_versions",
        "alembic_version"
    ]
    
    print(f"\nFound {len(tables)} tables:")
    for table in tables:
        status = "✅" if table in expected_tables else "⚠️"
        print(f"  {status} {table}")
    
    missing = [t for t in expected_tables[:-1] if t not in tables]  # Exclude alembic_version
    if missing:
        print(f"\n❌ Missing tables: {', '.join(missing)}")
        return False
    
    # Check table schemas
    print("\n3. Checking table schemas...")
    with engine.connect() as conn:
        for table in expected_tables[:-1]:  # Exclude alembic_version
            columns = inspector.get_columns(table)
            indexes = inspector.get_indexes(table)
            fks = inspector.get_foreign_keys(table)
            
            print(f"\n  Table: {table}")
            print(f"    Columns: {len(columns)}")
            print(f"    Indexes: {len(indexes)}")
            print(f"    Foreign Keys: {len(fks)}")
            
            # Check for UUID primary key
            has_uuid_pk = any(col['name'] == 'id' and 'UUID' in str(col['type']) for col in columns)
            print(f"    UUID PK: {'✅' if has_uuid_pk else '❌'}")
    
    # Check Alembic version
    print("\n4. Checking Alembic migration status...")
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version_num FROM alembic_version"))
        version = result.fetchone()
        if version:
            print(f"  ✅ Current migration: {version[0]}")
        else:
            print("  ❌ No migration applied")
            return False
    
    print("\n" + "=" * 70)
    print("✅ PHASE 3.7.1.2 DATABASE SETUP: COMPLETE")
    print("=" * 70)
    print(f"\nDatabase URL: {settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else 'N/A'}")
    print(f"Pool Size: {settings.DATABASE_POOL_SIZE}")
    print(f"Max Overflow: {settings.DATABASE_MAX_OVERFLOW}")
    
    return True

if __name__ == "__main__":
    success = verify_database()
    sys.exit(0 if success else 1)
