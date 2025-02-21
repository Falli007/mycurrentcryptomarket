from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import timedelta

app = Flask(__name__)

# Load pre-saved models into a dictionary.
# Adjust the filenames and keys as per your saved models.
models = {
    "BTC": joblib.load("model_BTC.joblib"),
    "ETH": joblib.load("model_ETH.joblib"),
    "LTC": joblib.load("model_LTC.joblib")
}

# Load your historical data once (or load per crypto as needed)
df = pd.read_csv("mycrypto.csv", parse_dates=["Date"])
df.sort_values("Date", inplace=True)

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        crypto = request.form.get("crypto")
        investment_str = request.form.get("investment", "0")
        investment = float(investment_str)

        # Filter data for selected crypto
        df_crypto = df[df["crypto_name"] == crypto].copy()
        df_crypto.sort_values("Date", inplace=True)
        
        # Create lagged feature if needed (here we use Close_lag1)
        df_crypto["Close_lag1"] = df_crypto["Close"].shift(1)
        df_crypto.dropna(inplace=True)
        
        # Get the last known close to seed the forecast
        last_close = df_crypto.iloc[-1]["Close"]
        
        # Retrieve the model for this crypto
        model = models[crypto]
        
        #Historical Predictions
        X_c = df_crypto[["Close_lag1"]]
        df_crypto["Predicted_Close"] = model.predict(X_c)
        
        
        # Generate future predictions (example: next 7 days)
        n_future = 7
        future_predictions = []
        for i in range(n_future):
            X_future = [[last_close]]
            pred_close = model.predict(X_future)[0]
            future_predictions.append(pred_close)
            last_close = pred_close
        
        # Generate future dates
        last_date = df_crypto["Date"].iloc[-1]
        future_dates = [last_date + timedelta(days=x) for x in range(1, n_future+1)]
        
        # Build a DataFrame for future predictions
        future_df = pd.DataFrame({
            "Date": future_dates,
            "Predicted_Close": future_predictions
        })
        
        # 5) Investment Calculation:
        #    Example approach: 
        #    - Assume user invests at day 1's predicted close => buys "num_coins" = investment / day1_price
        #    - The final value on day N is num_coins * dayN_price
        #    - We can show the potential gain or loss
        if len(future_df) > 0 and investment > 0:
            day1_price = future_df.iloc[0]["Predicted_Close"]  # price on the first future day
            dayN_price = future_df.iloc[-1]["Predicted_Close"] # price on the last future day
            
            # How many coins can the user buy with the initial investment?
            coins_bought = investment / day1_price
            
            # Potential final value after day N
            final_value = coins_bought * dayN_price
            
            # Potential profit or loss
            profit_loss = final_value - investment
        else:
            coins_bought = 0
            final_value = 0
            profit_loss = 0
        
        return render_template(
            "results.html",
            crypto=crypto,
            historical_data=df_crypto.to_dict(orient="records"),
            future_data=future_df.to_dict(orient="records"),
            investment=investment,
            coins_bought=coins_bought,
            final_value=final_value,
            profit_loss=profit_loss
        )
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
