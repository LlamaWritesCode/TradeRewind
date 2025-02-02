import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ai_trading_coach import ai_trading_coach
from data_fetcher import fetch_stock_data
from data_forecaster import forecast_with_prophet
from simulate_trade import simulate_trade_outcome

st.set_page_config(page_title="TradeRewind", layout="wide")
st.markdown("""
    <h1 style="text-align: center;">TradeRewind</h1> 
    <p style="text-align: center; font-size: 17px;">An AI Tool That Analyzes Your Trading Mistakes And Much More! </p>
""", unsafe_allow_html=True)

\
st.markdown("""
    <p style="text-align: center; font-size: 15px; margin-bottom: -20px;">
        Upload your trade history to analyze mistakes, explore alternative scenarios, 
        receive AI-powered forecasts, and get expert trading advice.
    </p>
""", unsafe_allow_html=True)
#################################### Trade Data File Upload: CSV ################################################## 
uploaded_file = st.file_uploader("", type=["csv"])

trade_history_records = []
if uploaded_file:
    trade_history = pd.read_csv(uploaded_file)
    required_columns = {"ticker", "buy_date", "sell_date", "sell_price", "allocation", "buy_price"}
    missing_columns = required_columns - set(trade_history.columns)
    if missing_columns:
        st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
    else:
        trade_history_records = trade_history.to_dict(orient="records")
        st.markdown("## 📜 Uploaded Trade History")
        st.dataframe(trade_history.style.set_properties(**{"background-color": "#272c36", "color": "white"}), height=300)

#################################### Popular Trading Strategies ################################################## 

with st.sidebar:
    st.markdown("## 📈 Trading Performance")

    if trade_history_records:
        valid_trades = [trade for trade in trade_history_records if 'sell_price' in trade and 'buy_price' in trade]
        
        if len(valid_trades) > 0:
            win_rate = (sum(1 for trade in valid_trades if trade['sell_price'] > trade['buy_price']) / len(valid_trades) * 100)
            total_profit_loss = sum(trade['sell_price'] - trade['buy_price'] for trade in valid_trades)
            avg_profit_per_trade = total_profit_loss / len(valid_trades) if len(valid_trades) > 0 else 0

            losing_trades = [trade for trade in valid_trades if trade['sell_price'] < trade['buy_price']]
            avg_loss_per_trade = (sum(trade['sell_price'] - trade['buy_price'] for trade in losing_trades) / len(losing_trades)) if len(losing_trades) > 0 else 1
            
            best_trade = max(valid_trades, key=lambda trade: (trade['sell_price'] - trade['buy_price']) / trade['buy_price'])
            worst_trade = min(valid_trades, key=lambda trade: (trade['sell_price'] - trade['buy_price']) / trade['buy_price'])

            risk_reward_ratio = abs(avg_profit_per_trade / avg_loss_per_trade) if avg_loss_per_trade != 0 else None

            # Custom CSS for smaller metric font
            st.markdown("""
                <style>
                    .metric-box {
                        text-align: center;
                        padding: 10px;
                        border-radius: 8px;
                        background-color: #1e1e1e;
                        border: 1px solid #444;
                        margin-bottom: 10px;
                    }
                    .metric-title {
                        font-size: 14px;
                        color: #fff;
                        font-weight: bold;
                    }
                    .metric-value {
                        font-size: 18px;
                        color: #4CAF50;
                        font-weight: bold;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Display metrics using custom HTML
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Trading Win Rate</div>
                        <div class="metric-value">{win_rate:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Best Trade Performance</div>
                        <div class="metric-value">{((best_trade['sell_price'] - best_trade['buy_price']) / best_trade['buy_price']) * 100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Total Profit/Loss</div>
                        <div class="metric-value">${total_profit_loss:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Worst Trade Performance</div>
                        <div class="metric-value">{((worst_trade['sell_price'] - worst_trade['buy_price']) / worst_trade['buy_price']) * 100:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Avg. Profit Per Trade</div>
                        <div class="metric-value">${avg_profit_per_trade:.2f}</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Risk-Reward Ratio</div>
                        <div class="metric-value">{risk_reward_ratio:.2f} </div>
                    </div>
                """, unsafe_allow_html=True)


    st.markdown("## 🔥 Risk & Money Management Metrics")
    st.markdown("- **Max Drawdown:** TBD%")
    st.markdown("- **Win/Loss Ratio:** TBD")
    st.markdown("- **Sharpe Ratio:** TBD")
    st.markdown("- **Portfolio Diversification Score:** TBD")


    st.markdown("## 📌 Popular Trading Strategies")
    strategy_descriptions = {
        "Momentum Trading": "📈 Buy assets that are rising and sell before momentum slows down.",
        "Mean Reversion": "🔄 Stocks tend to revert to their average price after big swings.",
        "Swing Trading": "📊 Hold stocks for days/weeks to capture short-term price movements.",
        "Day Trading": "⏳ Buy & sell within a single trading day to capitalize on short moves.",
        "Trend Following": "📉 Follow market trends—buy in an uptrend, sell in a downtrend.",
        "Breakout Trading": "🚀 Enter trades when price breaks key resistance/support levels.",
        "Scalping": "⚡ Make many quick trades for small profits throughout the day.",
        "Dollar-Cost Averaging": "📅 Invest a fixed amount over time to reduce market volatility.",
        "High-Risk Trading": "⚠️ Take aggressive trades with high reward potential but increased risk.",
        "Low-Risk Trading": "🛡 Focus on stable investments with lower risk and steady returns."
    }

    for rank, (strategy, description) in enumerate(strategy_descriptions.items(), 1):
        with st.expander(f"🔹 {rank}. {strategy}"):
            st.markdown(description)

    st.markdown("## 🔥 Risk Management Techniques")
    risk_management_tips = {
        "Position Sizing": "📏 Determine how much of your capital to allocate per trade to avoid overexposure.",
        "Stop-Loss Orders": "🚨 Automatically sell an asset when it reaches a set price to limit losses.",
        "Risk-Reward Ratio": "⚖️ Ensure potential profits outweigh risks—typically a 2:1 or 3:1 ratio is recommended.",
        "Diversification": "📊 Spread investments across different assets to minimize risk exposure.",
        "Trailing Stop-Loss": "📉 Adjusts stop-loss levels dynamically as prices move in your favor.",
        "Hedging Strategies": "🔄 Use options, futures, or other assets to offset potential losses.",
        "Avoid Over-Leveraging": "💰 Keep leverage low to prevent margin calls and excessive losses.",
        "Emotional Discipline": "🧠 Stick to a strategy and avoid impulsive trades based on fear or greed.",
        "Market Conditions Awareness": "🌍 Understand macroeconomic factors and market trends before trading.",
        "Risk Per Trade Rule": "🔢 Never risk more than 1-2% of your total capital on a single trade."
    }

    for rank, (technique, tip) in enumerate(risk_management_tips.items(), 1):
        with st.expander(f"🔹 {rank}. {technique}"):
            st.markdown(tip)


#################################### Stock Price Forecasting ################################################## 
if uploaded_file:
    st.markdown("## 🔮 AI-Powered Forecasting")
    st.write("Predict future stock prices based on past trends. This helps analyze potential missed gains or future trends for open trades.")

    forecast_results = []
    forecast_charts = {}

    for trade in trade_history_records:
        ticker = trade["ticker"]
        buy_date = trade["buy_date"]
        sell_date = trade["sell_date"]

        stock_prices = fetch_stock_data(ticker, buy_date)

        if stock_prices is not None:
            prophet_forecast = forecast_with_prophet(stock_prices)
            
            if prophet_forecast is not None:
                last_predicted_price = prophet_forecast.iloc[-1]["yhat"]
                if pd.isna(sell_date): 
                    forecast_results.append({"Ticker": ticker, "Predicted Price": round(last_predicted_price, 2)})
                else:  
                    profit_potential = round(last_predicted_price - trade["sell_price"], 2)
                    forecast_results.append({"Ticker": ticker, "Potential Missed Gain/Loss": f"${profit_potential}"})

               
                forecast_charts[ticker] = prophet_forecast

    if forecast_results:
        df_forecast = pd.DataFrame(forecast_results)

        st.markdown("### Forecasted Results")
        df_forecast["Predicted Price"] = pd.to_numeric(df_forecast["Predicted Price"], errors='coerce')
        df_forecast["Potential Missed Gain/Loss"] = df_forecast["Potential Missed Gain/Loss"].replace('[\$,]', '', regex=True).astype(float)
        st.dataframe(df_forecast.style.format({
            "Predicted Price": "${:.2f}",
            "Potential Missed Gain": "${:.2f}"
        }).set_table_styles(
            [{"selector": "thead th", "props": [("background-color", "#2b2b2b"), ("color", "white"), ("font-weight", "bold")]}]
        ).set_properties(**{"border": "1px solid white", "padding": "8px"}), height=250)

        selected_ticker = st.selectbox("Select a stock to view forecast trends:", forecast_charts.keys())
        if selected_ticker in forecast_charts:
            fig = px.line(forecast_charts[selected_ticker], x="ds", y="yhat", title=f"Predicted Price Movement for {selected_ticker}")
            st.plotly_chart(fig, use_container_width=True)


#################################### Trading Behaviour Analysis by Clustering ################################################## 
if uploaded_file:
    st.markdown("## 🤖 AI-Powered Trading Behavior Analysis")

def ml_trading_behavior_analysis(trades):
    if not trades or len(trades) < 3:
        return

    df = pd.DataFrame(trades)
    df["hold_duration"] = (pd.to_datetime(df["sell_date"], errors='coerce') - pd.to_datetime(df["buy_date"], errors='coerce')).dt.days
    df["profit_percent"] = ((df["sell_price"] - df["buy_price"]) / df["buy_price"]) * 100

    numeric_cols = ["profit_percent", "hold_duration", "allocation"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numeric_cols])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_features)

    cluster_summary = df.groupby("cluster")[numeric_cols].mean()
    dominant_cluster = df["cluster"].value_counts().idxmax()

    st.markdown(" 📊 Your Trading Behavior Cluster")
    st.info(f"🎯 Based on your trade data, you are categorized as: **Cluster {dominant_cluster}**")

    fig = px.bar(cluster_summary.T, barmode="group", title="Trading Behavior Per Cluster", labels={"index": "Metric", "value": "Average"}, text_auto=True)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Cluster Insights")

    cluster_labels = {
        0: "🚀 Aggressive Trader",
        1: "🛡 Conservative Trader",
        2: "📊 Balanced Trader"
    }

    cluster_advice = {
        0: "You take high risks with significant capital allocation. Consider using stop-loss strategies to limit potential losses.",
        1: "You adopt a cautious trading style with smaller capital allocations. You may be missing higher return opportunities.",
        2: "Your trades show a mix of risk and caution. Consider fine-tuning your allocation strategies for better diversification."
    }

    user_cluster = dominant_cluster  
    user_cluster_data = cluster_summary.loc[user_cluster]

    st.markdown(f"""
        <div style="border-radius:12px; padding:15px; background-color:#1e1e1e; color:white; border: 2px solid #ddd; text-align: center;">
            <h3>{cluster_labels[user_cluster]}</h3>
            <p style="font-size: 18px; color: #4CAF50 if user_cluster == 0 else '#FF9800' if user_cluster == 1 else '#2196F3';"><b>🏷 Cluster ID:</b> {user_cluster}</p>
            <p><b>📈 Avg. Profit %:</b> <span style="color: {'#4CAF50' if user_cluster_data['profit_percent'] > 0 else '#FF5733'};">{user_cluster_data["profit_percent"]:.2f}%</span></p>
            <p><b>⏳ Avg. Hold Duration:</b> {user_cluster_data["hold_duration"]:.2f} days</p>
            <p><b>💰 Avg. Allocation:</b> {user_cluster_data["allocation"]:.2f}%</p>
            <p style="background-color:rgba(255,255,255,0.1); padding:10px; border-radius:8px;"><b>🔹 Personalized Advice:</b> {cluster_advice[user_cluster]}</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("Your Cluster Data Breakdown")

    cluster_summary_renamed = cluster_summary.rename(columns={
        "profit_percent": "Avg. Profit (%)",
        "hold_duration": "Avg. Hold Duration (days)",
        "allocation": "Avg. Allocation (%)"
    })

    st.write(cluster_summary_renamed.style.format({
        "Avg. Profit (%)": "{:.2f}%",
        "Avg. Hold Duration (days)": "{:.1f} days",
        "Avg. Allocation (%)": "{:.1f}%"
    }).set_table_styles(
        [{"selector": "thead th", "props": [("background-color", "#2b2b2b"), ("color", "white"), ("font-weight", "bold")]}]
    ).set_properties(**{"border": "1px solid white", "padding": "8px"}))


ml_trading_behavior_analysis(trade_history_records)


#################################### What If Scenarios ################################################## 
st.markdown("## 🔍 What-If Scenarios")
st.write("Modify hold duration or sell price to see how different decisions would have impacted your profits.")


if trade_history_records:
    trade_options = [
        f"{trade['ticker']} | {trade['buy_date']} → {trade['sell_date']} | Sell: ${trade['sell_price']} | Allocation: {trade['allocation']}%"
        for trade in trade_history_records
    ]
    
    selected_trade_index = st.selectbox("Select a trade to simulate:", range(len(trade_options)), format_func=lambda x: trade_options[x])
    selected_trade = trade_history_records[selected_trade_index]

    col1, col2 = st.columns(2)
    with col1:
        new_hold_days = st.slider("Adjust Hold Duration (Days)", min_value=-30, max_value=60, value=0)

    with col2:
        new_sell_price = st.slider(
            "Adjust Sell Price ($)",
            min_value=float(selected_trade['buy_price'] * 0.8),  
            max_value=float(selected_trade['buy_price'] * 1.5), 
            value=float(selected_trade['sell_price']) 
        )

    new_estimated_sell_price, profit_diff = simulate_trade_outcome(selected_trade, new_hold_days, new_sell_price)

    if new_estimated_sell_price:
        st.success(f"📊 **New Estimated Sell Price:** ${new_estimated_sell_price}")
        st.success(f"💰 **Profit Difference:** {'+' if profit_diff > 0 else ''}${profit_diff}")

    else:
        st.error(f"❌ Unable to calculate. {profit_diff}")


#################################### AI Powered Trading Tips ################################################## 
st.markdown("## 📢 AI Trading Tips")
advice = ai_trading_coach(trade_history_records)
st.info(f"💡 {advice}")
