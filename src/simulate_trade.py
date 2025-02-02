import pandas as pd

def simulate_trade_outcome(trade, new_hold_duration, new_sell_price):
    try:
        buy_date = pd.to_datetime(trade['buy_date'])
        sell_date = pd.to_datetime(trade['sell_date'])

        original_hold_duration = (sell_date - buy_date).days
        if original_hold_duration <= 0:
            original_hold_duration = 1  

        price_growth = (trade["sell_price"] - trade["buy_price"]) / trade["buy_price"]
        daily_growth_rate = price_growth / original_hold_duration

        adjusted_sell_price = trade["buy_price"] * (1 + daily_growth_rate * new_hold_duration)

        profit_diff = adjusted_sell_price - trade["sell_price"]

        return round(adjusted_sell_price, 2), round(profit_diff, 2)
    
    except Exception as e:
        return None, f"Error: {str(e)}"