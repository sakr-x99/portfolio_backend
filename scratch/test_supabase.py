import os
import sys

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.modules.blog.storage import upload_markdown_file

def test_upload():
    sample_content = b"# Test Markdown\nThis is a test file for Supabase Storage."
    filename = "test_article.md"
    
    print(f"Attempting to upload {filename}...")
    url = upload_markdown_file(sample_content, filename)
    
    if url:
        print(f"Success! Public URL: {url}")
    else:
        print("Upload failed. Check your Supabase credentials in .env and ensure the bucket exists.")

if __name__ == "__main__":
    test_upload()
