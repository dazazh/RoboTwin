from openai import OpenAI

# Configure the API and key (using DeepSeek as an example)
def generate(message, gpt="deepseek"):

    if gpt == "deepseek":
        MODEL = "deepseek-chat"
        OPENAI_API_BASE="https://api.deepseek.com"
        # Set your API key here
        OPENAI_API_KEY="sk-0bc806156bb04622817a392d809b92e9"

    client = OpenAI(api_key=OPENAI_API_KEY, base_url="https://api.deepseek.com")
    print('start generating')
    response = client.chat.completions.create(
        model=MODEL,
        messages=message,
        stream=False,
        temperature=0,
    )
    print('end generating')

    return response.choices[0].message.content 
