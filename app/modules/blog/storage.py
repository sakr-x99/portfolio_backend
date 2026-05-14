import os
from typing import Optional
from app.core.supabase_client import supabase
from app.core.config import settings

def upload_markdown_file(file_content: bytes, filename: str, bucket: str = None) -> Optional[str]:
    """
    Uploads a markdown file to Supabase Storage and returns the public URL.
    """
    if not supabase:
        print("Supabase client not initialized. Check your credentials.")
        return None
    
    target_bucket = bucket or settings.SUPABASE_BUCKET
    
    try:
        # Ensure unique filename if needed, or use provided
        # path could be "articles/filename.md"
        path = f"articles/{filename}"
        
        # Upload file
        response = supabase.storage.from_(target_bucket).upload(
            path=path,
            file=file_content,
            file_options={"upsert": "true", "content-type": "text/markdown"}
        )
        
        # Get public URL
        public_url = supabase.storage.from_(target_bucket).get_public_url(path)
        return public_url
        
    except Exception as e:
        print(f"Error uploading to Supabase: {str(e)}")
        return None

def delete_markdown_file(path: str, bucket: str = None):
    """
    Deletes a file from Supabase Storage.
    """
    if not supabase:
        return
    
    target_bucket = bucket or settings.SUPABASE_BUCKET
    try:
        supabase.storage.from_(target_bucket).remove([path])
    except Exception as e:
        print(f"Error deleting from Supabase: {str(e)}")
