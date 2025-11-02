# QuantVista
A python based trading suggestion or analysis webapp dashboard that fetches api calls in realtime.
# AngelOne Real-time Options Dashboard

A high-performance, low-latency trading dashboard for NSE indices (NIFTY, BANKNIFTY, FINNIFTY), providing real-time technical signals, options greeks, and open interest (OI) analysis via the Angel One Smart API.


<img width="1920" height="1080" alt="Screenshot (97)" src="https://github.com/user-attachments/assets/192e156c-695f-447a-bece-3b3ee77ac696" />


---

## ⚡ Core Features

* **Real-time Index Data:** Live data for NIFTY, BANKNIFTY, and FINNIFTY.
* **Low-Latency:** Sub-3-second data refresh using a FastAPI WebSocket backend.
* **Advanced Options Analysis:** Fetches and displays the full options chain with:
    * **Real-time Greeks:** Delta, Gamma, Theta, and Vega.
    * **Real-time OI:** Open Interest for every strike.
    * **Real-time Volume:** Live volume for every strike.
* **Portfolio Greeks Summary:** Calculates and displays key metrics like Net Delta, Put-Call Ratio (PCR), and Max Pain for the entire expiry.
* **Technical Signal Engine:** Generates signals from 12 distinct indicators (RSI, MACD, Moving Averages, Bollinger Bands, etc.).
* **Consolidated Final Signal:** Provides a single, scannable **BUY/SELL/HOLD** recommendation with a confidence score, entry price, stop-loss, and targets.
* **Market Intelligence:** Includes a real-time market status (Open/Closed/Pre-Market) and a full NSE trading calendar with holidays.

## 🛠️ Tech Stack

* **Backend:** **Python 3.10+**, **FastAPI**, **Uvicorn**
* **API:** **Angel One Smart API** (using `requests` and `smartapi-python`)
* **Frontend:** **HTML5**, **Tailwind CSS**, **Vanilla JavaScript** (WebSocket client)
* **Data Analysis:** `pandas`, `numpy`
* **Database:** `sqlite3` (for logging signals)



## 🏗️ Architecture (How it Works)

This dashboard is built for speed and low latency by efficiently minimizing API calls.

1.  The FastAPI backend logs into the Angel One Smart API on startup.
2.  When you open the dashboard, the frontend establishes a WebSocket connection for each index (e.g., `/ws/NIFTY`).
3.  The backend starts a high-speed loop for that index (e.g., every 2 seconds).
4.  **In just two (2) API calls**, the backend fetches all necessary options data:
    * **Call 1 (Greeks):** A direct `requests.post` call to the `optionGreek` API endpoint to get all Greeks (Delta, Theta, etc.) for the entire weekly expiry.
    * **Call 2 (OI & LTP):** A single `getMarketData` call to get the real-time OI, Volume, and LTP for all the tokens fetched in the first call.
5.  The backend then calculates all 12 technical indicators, merges all data, and generates the final signal.
6.  The complete JSON payload is pushed to your frontend via the WebSocket, where JavaScript updates the dashboard.



