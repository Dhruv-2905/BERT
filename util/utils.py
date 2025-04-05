import json

def load_json_data(file_path):
    """Load JSON data from a file"""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    print(f"✅ JSON loaded! Total records: {len(data['data'])}")
    return data["data"]

def extract_passages(data):
    """Extract contexts from JSON"""
    passages = [para["context"] for entry in data for para in entry.get("paragraphs", [])]
    print(f"✅ Extracted {len(passages)} passages")
    return passages
