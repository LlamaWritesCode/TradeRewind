import openai
import config

openai.api_key = config.OPENAI_KEY

def ai_trading_coach(mistakes):
    prompt = f"You are a trading coach. The user made these mistakes: {mistakes}. Provide trading advice."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]
    