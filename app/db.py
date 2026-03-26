from pymongo import MongoClient
from pymongo.collection import Collection
from app.config import settings

client = MongoClient(settings.mongodb_uri)
db = client[settings.mongodb_db]
collection: Collection = db[settings.mongodb_collection]