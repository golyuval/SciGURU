import os
from dotenv import load_dotenv
from openai import OpenAI

def test_api_key():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return False
    
    try:
        # Initialize the client
        client = OpenAI(api_key=api_key)
        
        # Make a simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=5
        )
        
        print("✅ API key test successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_api_key() 