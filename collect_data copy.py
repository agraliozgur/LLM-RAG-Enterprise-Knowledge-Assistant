import google.generativeai as genai
import json

# API anahtarınızı buraya ekleyin (https://aistudio.google.com/app/apikey)
API_KEY = "AIzaSyCgaZ-ZUqvBumLwvVT7naSUOpmWYy6PaBE"
MODEL_NAME = "gemini-1.5-pro-latest"

# Configure the model
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def generate_ekm_synthetic_data(prompt, num_examples=5):
    examples = []
    
    system_prompt = """You are an Enterprise Knowledge Management expert. Generate realistic but completely synthetic data in this structure:
    - Company information (name, industry, employee count)
    - Tools and technologies used
    - Key challenges faced
    - Implemented solutions
    - Measured outcomes
    All data must be fictional!"""
    
    full_prompt = f"{system_prompt}\n\n{prompt}\nNumber of examples: {num_examples}"
    
    try:
        response = model.generate_content(full_prompt)
        
        if response.text:
            generated_data = []
            # Convert generated text to structured format
            for example in response.text.split("\n\n"):
                if example.strip():
                    data_point = {}
                    lines = example.split("\n")
                    for line in lines:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            data_point[key.strip()] = value.strip()
                    generated_data.append(data_point)
            return generated_data
        return []
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

# Example usage
if __name__ == "__main__":
    prompt = """Create a synthetic case study about improving knowledge sharing processes 
    for a global retail company"""
    
    synthetic_data = generate_ekm_synthetic_data(prompt, num_examples=3)
    
    # Save as JSON
    with open("synthetic_ekm_data.json", "w", encoding="utf-8") as f:
        json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
    
    print("Data saved successfully!")