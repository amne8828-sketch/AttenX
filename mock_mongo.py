import uuid
from datetime import datetime

class MockCollection:
    def __init__(self, name):
        self.name = name
        self.data = {}  # Map _id to document

    def insert_one(self, document):
        if "_id" not in document:
            document["_id"] = str(uuid.uuid4())
        self.data[document["_id"]] = document
        return type('InsertOneResult', (object,), {"inserted_id": document["_id"]})()

    def find_one(self, query=None):
        if not query:
            return list(self.data.values())[0] if self.data else None
        
        for doc in self.data.values():
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    def find(self, query=None):
        results = []
        if not query:
            return MockCursor(list(self.data.values()))
        
        for doc in self.data.values():
            match = True
            for k, v in query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                results.append(doc)
        return MockCursor(results)

    def update_one(self, query, update):
        doc = self.find_one(query)
        if doc:
            if "$set" in update:
                doc.update(update["$set"])
            return type('UpdateResult', (object,), {"modified_count": 1})()
        return type('UpdateResult', (object,), {"modified_count": 0})()

    def delete_one(self, query):
        doc = self.find_one(query)
        if doc:
            del self.data[doc["_id"]]
            return type('DeleteResult', (object,), {"deleted_count": 1})()
        return type('DeleteResult', (object,), {"deleted_count": 0})()
        
    def count_documents(self, query=None):
        if not query:
            return len(self.data)
        return len(list(self.find(query)))

class MockCursor(list):
    def sort(self, *args, **kwargs):
        return self
    def limit(self, *args, **kwargs):
        return self

class MockDatabase:
    def __init__(self, name):
        self.name = name
        self.collections = {}

    def __getitem__(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection(name)
        return self.collections[name]
        
    @property
    def command(self):
        def _command(*args, **kwargs):
            return {"ok": 1}
        return _command

class MockMongoClient:
    def __init__(self, uri=None, **kwargs):
        self.databases = {}
        print(f"⚠️ Using Mock MongoDB Client (In-Memory Only)")

    def __getitem__(self, name):
        if name not in self.databases:
            self.databases[name] = MockDatabase(name)
        return self.databases[name]
        
    @property
    def admin(self):
        return self['admin']
