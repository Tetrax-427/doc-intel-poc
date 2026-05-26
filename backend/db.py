import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

def insert_document(name: str) -> str:
    result = supabase.table("documents").insert({"name": name}).execute()
    return result.data[0]["id"]

def insert_chunks(chunks: list[dict]):
    supabase.table("chunks").insert(chunks).execute()

def get_all_chunks() -> list[dict]:
    result = supabase.table("chunks").select("*").execute()
    
    return result.data