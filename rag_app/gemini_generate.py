import google.generativeai as genai
from config import GOOGLE_API_KEY

def generate_response(prompt):
    genai.configure(api_key="AIzaSyAVch1PrXWTDMk5k2SwgRH4jHVUtWzaUjw")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
