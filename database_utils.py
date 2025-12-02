import os
from pymongo import MongoClient
import datetime

# Global database client
db_client = None

def get_db_collections():
    """
    Establishes (if needed) and returns a dictionary of MongoDB collections.
    Also returns the raw client if needed.
    """
    global db_client
    
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    
    if db_client is None:
        try:
            client = MongoClient(MONGODB_URI, 
                               serverSelectionTimeoutMS=15000,
                               connectTimeoutMS=15000,
                               socketTimeoutMS=15000,
                               retryWrites=False)
            # Verify connection
            client.admin.command('ping')
            print("MongoDB connection successful!")
            db_client = client
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            try:
                client = MongoClient(MONGODB_URI, 
                                   serverSelectionTimeoutMS=15000,
                                   tls=False,
                                   retryWrites=False)
                client.admin.command('ping')
                print("MongoDB connection successful with TLS disabled!")
                db_client = client
            except Exception as e2:
                print(f"All MongoDB attempts failed: {e2}")
                try:
                    from mock_mongo import MockMongoClient
                    client = MockMongoClient(MONGODB_URI)
                    print("Mock MongoDB initialized!")
                    db_client = client
                except ImportError:
                    # Last resort fallback
                    db_client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    
    # Use the global client
    client = db_client
    
    transactional_db = client["transactional_db"]
    attendance_col = transactional_db["attendance"]
    
    core = client['secure_db']
    persons_col = core["persons"]
    profile_col = core["profile"]
    superadmins_col = core["superadmins"]
    admins_col = core["admins"]
    users_col = core["users"]
    enrollment_requests_col = core["enrollment_requests"]
    system_logs_col = core["system_logs"]
    cameras_col = core["cameras"]
    
    return {
        'client': client,
        'attendance_col': attendance_col,
        'persons_col': persons_col,
        'profile_col': profile_col,
        'superadmins_col': superadmins_col,
        'admins_col': admins_col,
        'users_col': users_col,
        'enrollment_requests_col': enrollment_requests_col,
        'system_logs_col': system_logs_col,
        'cameras_col': cameras_col
    }
