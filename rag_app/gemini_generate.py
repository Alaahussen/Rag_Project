import google.generativeai as genai
from config import GOOGLE_API_KEY

def generate_response(prompt):
    genai.configure(api_key="AIzaSyAT_W941pudYmJZid8gE2jitqI2VS1teZs")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
