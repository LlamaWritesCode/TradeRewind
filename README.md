## **📜 TradeRewind - AI-Powered Trading Assistant**


🚀 **TradeRewind** is an AI-powered trading assistant that helps traders **analyze mistakes**, **predict stock prices**, **simulate alternative scenarios**, and **receive expert trading insights**. Upload your trade history and let TradeRewind do the rest!

* * * * *

**✨ Features**
--------------

✅ **Trading Performance Metrics**: Win rate, profit/loss, risk-reward ratio, and more.\
✅ **AI-Powered Stock Price Forecasting**: Predict future stock trends using historical data.\
✅ **Trading Behavior Analysis**: Uses AI clustering to categorize your trading style.\
✅ **What-If Scenario Simulator**: See how different trade decisions could have affected your profits.\
✅ **Risk & Money Management Insights**: Learn best practices for minimizing risk.\
✅ **AI Trading Coach**: Get AI-driven trading advice based on your history.

* * * * *


**🛠 Tech Stack**
-----------------

| Component | Technology/Library |
| --- | --- |
| **Frontend** | Streamlit UI |
| **Machine Learning** | Scikit-learn, Prophet (Forecasting) |
| **Data Processing** | Pandas, NumPy |
| **Stock Data** | Yahoo Finance API |
| **Backend** | Python, Flask (Optional) |
| **Hosting** | Streamlit Cloud, Hugging Face Spaces |

* * * * *

**🚀 Installation & Setup**
---------------------------

### **Step 1: Clone the Repository**

`git clone https://github.com/LlamaWritesCode/money-bee.git
cd money-bee`

### **Step 2: Create a Virtual Environment**
`python -m venv venv
source venv/bin/activate   # MacOS/Linux
venv\Scripts\activate      # Windows`

### **Step 3: Install Dependencies**
`pip install -r requirements.txt`

### **Step 4: Set Up API Keys**

1️⃣ **Create a `.env` file in the `src/` directory**:

`touch src/.env`

2️⃣ **Add your API key inside**:

`API_KEY=your_secret_api_key_here`

### **Step 5: Run the App**

`streamlit run src/app.py`

🎉 The app should now be running at `http://localhost:8501`

* * * * *


**💡 How It Works**
-------------------

1️⃣ **Upload your trade history (`CSV`)**.\
2️⃣ **Analyze** mistakes, profits, and trading behavior.\
3️⃣ **Get AI-powered forecasts** for stock trends.\
4️⃣ **Run "What-If" scenarios** to test different trade strategies.\
5️⃣ **Receive AI Trading Coach insights** to improve your decisions.

* * * * *

## 📌 Example Trade History CSV Format
---------------------------------------

| Ticker | Buy Date   | Sell Date  | Buy Price | Sell Price | Allocation |
|--------|-----------|------------|-----------|------------|------------|
| AAPL   | 2024-01-01 | 2024-01-10 | 150       | 165        | 20%        |
| GOOGL  | 2024-02-05 | 2024-02-15 | 2800      | 2750       | 30%        |



**📜 License**
--------------

📝 MIT License - Feel free to use, modify, and distribute!

* * * * *

**👨‍💻 Author**
----------------

**LlamaWritesCode**\
🔗 [GitHub](https://github.com/LlamaWritesCode)
