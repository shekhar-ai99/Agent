# import os
# from dotenv import load_dotenv
# import openai

# # ✅ Load environment variables BEFORE accessing them
# load_dotenv()

# # ✅ Now this will work
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     raise ValueError("OPENAI_API_KEY not found in environment variables.")

# # Sample input data
# current_row = {
#     'ema_9': 220.5,
#     'ema_50': 215.7,
#     'macd': 1.2,
#     'macd_signal': 0.9,
#     'macd_hist': 0.3,
#     'rsi': 60,
#     'plus_di': 25,
#     'minus_di': 10,
#     'adx': 30,
#     'close': 222.0
# }

# # Set up OpenAI client
# client = openai.OpenAI(api_key=api_key)

# # Build the prompt
# prompt = f"""
# You are a trading assistant. Based on the following indicator values, suggest whether to BUY, SELL, or HOLD:

# - EMA(9): {current_row['ema_9']}
# - EMA(50): {current_row['ema_50']}
# - MACD: {current_row['macd']}
# - MACD Signal: {current_row['macd_signal']}
# - MACD Histogram: {current_row['macd_hist']}
# - RSI: {current_row['rsi']}
# - +DI: {current_row['plus_di']}
# - -DI: {current_row['minus_di']}
# - ADX: {current_row['adx']}
# - Close Price: {current_row['close']}

# Respond with only one word: BUY, SELL, or HOLD.
# """

# # Make API call
# response = client.chat.completions.create(
#     model="gpt-4",
#     messages=[
#         {"role": "system", "content": "You are a trading strategy decision assistant."},
#         {"role": "user", "content": prompt}
#     ]
# )

# # Get the signal
# signal = response.choices[0].message.content.strip().upper()
# print(f"Signal: {signal}")
import sklearn
import xgboost
import lightgbm

print("✅ All libraries imported successfully!")
