import openai
import os

def ai_trading_coach(trade_history_records):
    openai_api_key = os.getenv("OPENAI_KEY")

    if not openai_api_key:
        return "❌ OpenAI API key is missing. Please set it in your environment variables."

    client = openai.OpenAI(api_key=openai_api_key)

    response = client.chat.completions.create(
        model="gpt-4", 
        messages=[
            {"role": "system", "content": "You are an AI trading coach providing expert financial insights."},
            {"role": "user", "content": f"Analyze this trade history and provide suggestions: {trade_history_records}"}
        ]
    )

    return response.choices[0].message.content.strip()
