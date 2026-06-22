from dotenv import load_dotenv
load_dotenv()

from google import genai

client = genai.Client()  # reads GEMINI_API_KEY from the environment

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly: connection works",
)
print(response.text)

