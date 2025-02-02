import pandas as pd
from prophet import Prophet

def forecast_with_prophet(stock_prices):
    if len(stock_prices) < 30:
        return None

    df = pd.DataFrame({"ds": stock_prices.index, "y": stock_prices.values.ravel()}).reset_index(drop=True)

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]]