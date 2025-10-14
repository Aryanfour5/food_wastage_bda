import csv
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client.flw
coll = db.records

def to_int(val):
    try: return int(val)
    except: return None

def to_float(val):
    try: return float(val)
    except: return None

with open("loader/Data.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    batch = []
    for row in reader:
        row["m49_code"] = to_int(row.get("m49_code"))
        row["year"] = to_int(row.get("year"))
        row["loss_percentage"] = to_float(row.get("loss_percentage"))
        row["loss_percentage_original"] = row.get("loss_percentage_original")
        row["loss_quantity"] = to_float(row.get("loss_quantity"))
        row["sample_size"] = to_float(row.get("sample_size"))
        for key in row:
            if row[key] == "":
                row[key] = None
        batch.append(row)
        if len(batch) >= 5000:
            coll.insert_many(batch)
            batch = []
    if batch:
        coll.insert_many(batch)

coll.create_index([("year", 1), ("commodity", 1), ("country", 1)])
coll.create_index([("food_supply_stage", 1), ("cause_of_loss", 1)])
print("Ingestion & Indexing finished.")
