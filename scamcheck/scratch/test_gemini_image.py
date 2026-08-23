import os
from PIL import Image
import io
from google import genai
from google.genai import types

from backend.config import get_gemini_api_key

api_key = get_gemini_api_key()
client = genai.Client(api_key=api_key)

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=img_bytes.read(), mime_type="image/png"),
            "What color is this image?",
        ],
    )
    print("SUCCESS")
    print(response.text)
except Exception as e:
    print(f"FAILED: {e}")
