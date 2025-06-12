import google.generativeai as genai
#from config import GOOGLE_API_KEY

def generate_response(prompt):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
