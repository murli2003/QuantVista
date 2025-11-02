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
Here are some name suggestions for your project, followed by a complete README.md file for your GitHub repository.

💡 Project Name Suggestions
Here are a few good names, categorized by what they emphasize:

Descriptive (Clear & Searchable):

AngelOne-OI-Greeks-Dashboard

NSE-Options-Dashboard

AngelOne-Signal-Engine

Brandable (Short & Catchy):

QuantVista

NSE-Pulse

GreeksEye

AngelEdge

Technical (Focus on the Tech):

PySignal-Dash

FastAPI-Options-Trader

🚀 GitHub README File
Here is a complete, professional README.md file. You can copy and paste this directly into a file named README.md in your project folder.

Markdown

# AngelOne Real-time Options Dashboard

A high-performance, low-latency trading dashboard for NSE indices (NIFTY, BANKNIFTY, FINNIFTY), providing real-time technical signals, options greeks, and open interest (OI) analysis via the Angel One Smart API.

![Dashboard Demo](httpsDASHBOARD_IMAGE_LINK_HERE)
*(Suggestion: Upload a screenshot of your dashboard and replace the link above)*

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

## 🚀 Getting Started

Follow these steps to get the dashboard running on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git](https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git)
cd YOUR_PROJECT_NAME
2. Install Dependencies
It's highly recommended to use a Python virtual environment.

Bash

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from requirements.txt
pip install -r requirements.txtHere are some name suggestions for your project, followed by a complete README.md file for your GitHub repository.

💡 Project Name Suggestions
Here are a few good names, categorized by what they emphasize:

Descriptive (Clear & Searchable):

AngelOne-OI-Greeks-Dashboard

NSE-Options-Dashboard

AngelOne-Signal-Engine

Brandable (Short & Catchy):

QuantVista

NSE-Pulse

GreeksEye

AngelEdge

Technical (Focus on the Tech):

PySignal-Dash

FastAPI-Options-Trader

🚀 GitHub README File
Here is a complete, professional README.md file. You can copy and paste this directly into a file named README.md in your project folder.

Markdown

# AngelOne Real-time Options Dashboard

A high-performance, low-latency trading dashboard for NSE indices (NIFTY, BANKNIFTY, FINNIFTY), providing real-time technical signals, options greeks, and open interest (OI) analysis via the Angel One Smart API.

![Dashboard Demo](httpsDASHBOARD_IMAGE_LINK_HERE)
*(Suggestion: Upload a screenshot of your dashboard and replace the link above)*

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

## 🚀 Getting Started

Follow these steps to get the dashboard running on your local machine.

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git](https://github.com/YOUR_USERNAME/YOUR_PROJECT_NAME.git)
cd YOUR_PROJECT_NAME
2. Install Dependencies
It's highly recommended to use a Python virtual environment.

Bash

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install from requirements.txt
pip install -r requirements.txt
Note: The included smartapi-python==1.3.0 is old. It's recommended to upgrade it for better stability: pip install smartapi-python --upgrade

3. Configure Your API Credentials
This is the most important step. Open backend.py with any text editor.

Find the CONFIG section at the top of the file (around line 19) and enter your Angel One Smart API credentials.

Python

# ----- CONFIG -----
API_KEY = "YOUR_API_KEY_HERE"
CLIENT_CODE = "YOUR_CLIENT_ID"
PASSWORD = "YOUR_LOGIN_PIN"
TOTP_SECRET = "YOUR_TOTP_SECRET_KEY" # This is the long secret key, not the 6-digit number
4. Run the Backend Server
Once configured, run the FastAPI server from your terminal:

Bash

python backend.py
You should see Uvicorn start the server on port 8000:

INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Angel One TOTP: 123456
INFO:     LOGIN RESPONSE: {'status': True, ...}
INFO:     Feed token obtained: ...
INFO:     Angel One API connected successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on [http://0.0.0.0:8000](http://0.0.0.0:8000) (Press CTRL+C to quit)
5. Open the Dashboard
Open your web browser and go to:

http://127.0.0.1:8000

The dashboard will load, connect to the WebSocket, and start streaming real-time data.
