import os
import logging
import pytz
import schedule
import time
import threading
from datetime import datetime, timedelta
import asyncio
import nest_asyncio

import pandas as pd
import numpy as np
import requests

# OANDA API
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API

# Telegram Bot
from telegram import Bot
from telegram.ext import ApplicationBuilder

# FastAPI
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Environment Variables
from dotenv import load_dotenv
load_dotenv()

# Apply nest_asyncio to handle event loop issues
nest_asyncio.apply()

class ForexLiveTradeBot:
    def __init__(self):
        # Logging setup
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Trading Pairs
        self.pairs = [
            'EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 
            'GBP_USD', 'GBP_JPY', 'XAU_USD', 'EUR_JPY', 
            'AUD_JPY', 'USD_CHF', 'NZD_USD', 'EUR_GBP', 
            'EUR_AUD', 'GBP_AUD', 'AUD_NZD', 'GBP_NZD'
        ]

        # Add default timeframe
        self.timeframe = 'M30' 

        # Risk Management
        self.risk_amount = 100  # Default risk per trade
        self.min_risk_reward = 1.75  # Minimum risk-reward ratio

        # OANDA Configuration
        self.oanda_api_key = os.getenv('OANDA_API_KEY')
        
        if not self.oanda_api_key:
            raise ValueError("OANDA API credentials are required")
        
        self.oanda_client = API(access_token=self.oanda_api_key)

        # Create a single event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Telegram Configuration
        self.telegram_token = os.getenv('TELEGRAM_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.telegram_bot = None
        
        if self.telegram_token and self.telegram_chat_id:
            try:
                # Initialize Telegram Bot
                self.telegram_bot = Bot(token=self.telegram_token)
                self.logger.info("Telegram bot initialized successfully")
                
                # Send startup message using loop
                self.loop.run_until_complete(self.send_telegram_signal({
                    'pair': 'SYSTEM',
                    'type': 'STARTUP',
                    'entry_price': 0,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'custom_message': f"🤖 Trading Bot Started\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTrading Pairs: {', '.join(self.pairs)}"
                }))
            except Exception as e:
                self.logger.error(f"Failed to initialize Telegram bot: {e}")
                self.telegram_bot = None

    async def send_telegram_signal(self, signal: dict):
        """
        Asynchronously send trade signal to Telegram
        """
        if not self.telegram_bot:
            return
        
        try:
            # Check if there's a custom message
            if 'custom_message' in signal:
                message = signal['custom_message']
            else:
                message = f"""🚨 NEW TRADE SIGNAL 🚨
Pair: {signal['pair']}
Type: {signal['type']}
Entry Price: {signal['entry_price']:.4f}
Stop Loss: {signal['stop_loss']:.4f}
Take Profit: {signal['take_profit']:.4f}
Risk Reward: {self.min_risk_reward}:1
"""
            
            # Send message asynchronously
            async def send_message():
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id, 
                    text=message
                )
        except Exception as e:
            self.logger.error(f"Error sending Telegram message: {str(e)}")

    def sync_send_telegram_signal(self, signal: dict):
        """
        Synchronous wrapper for async send_telegram_signal
        """
        if self.telegram_bot:
            # Use the instance's event loop
            self.loop.run_until_complete(self.send_telegram_signal(signal))

    def fetch_historical_data(self, pair: str, count: int = 500) -> pd.DataFrame:
        """
        Fetch historical price data for a given currency pair
        """
        try:
            params = {
                "count": count,
                "granularity": self.timeframe,
                "price": "M"  # Midpoint pricing
            }
            
            r = instruments.InstrumentsCandles(
                instrument=pair,
                params=params
            )
            
            self.oanda_client.request(r)
            
            prices = []
            for candle in r.response['candles']:
                if candle['complete']:
                    prices.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c'])
                    })
            
            df = pd.DataFrame(prices)
            df.set_index('time', inplace=True)
            df['pair'] = pair
            
            return df

        except Exception as e:
            self.logger.error(f"Error fetching data for {pair}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        """
        df = data.copy()
        
        # EMA Calculations
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # MACD Calculation
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']
        
        # RSI Calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ADX Calculation
        high, low, close = df['high'], df['low'], df['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        df['ADX'] = dx.rolling(14).mean()
        
        return df.dropna()

    def find_trade_signals(self, data: pd.DataFrame) -> dict:
        """
        Find potential trade signals based on strategy rules
        """
        current = data.iloc[-1]
        pair = data['pair'].iloc[0]
        
        # Long Entry Conditions
        long_conditions = (
            current['EMA50'] > current['EMA200'] and
            current['MACD_Hist'] > 0 and
            current['RSI'] < 65 and
            current['ADX'] > 25
        )
        
        # Short Entry Conditions
        short_conditions = (
            current['EMA50'] < current['EMA200'] and
            current['MACD_Hist'] < 0 and
            current['RSI'] > 35 and
            current['ADX'] > 25
        )
        
        # Determine trade direction and details
        if long_conditions:
            return {
                'pair': pair,
                'type': 'LONG',
                'entry_price': current['close'],
                'stop_loss': current['low'],
                'take_profit': current['close'] + (current['close'] - current['low']) * self.min_risk_reward
            }
        
        if short_conditions:
            return {
                'pair': pair,
                'type': 'SHORT',
                'entry_price': current['close'],
                'stop_loss': current['high'],
                'take_profit': current['close'] - (current['high'] - current['close']) * self.min_risk_reward
            }
        
        return None

    def scan_markets(self, specific_pairs=None):
        """
        Scan all market pairs for trade signals
        """
        scan_pairs = specific_pairs or self.pairs
        signals = []
        
        for pair in scan_pairs:
            try:
                # Fetch historical data
                data = self.fetch_historical_data(pair)
                
                if data.empty:
                    self.logger.warning(f"No data for {pair}")
                    continue
                
                # Calculate indicators
                indicators_data = self.calculate_indicators(data)
                
                # Find trade signal
                signal = self.find_trade_signals(indicators_data)
                
                if signal:
                    signals.append(signal)
                    # Use sync_send_telegram_signal
                    self.sync_send_telegram_signal(signal)
                
            except Exception as e:
                self.logger.error(f"Error processing {pair}: {str(e)}")
        
        return signals

    def run(self):
        """
        Main execution method with improved event loop management
        """
        self.logger.info("Trading Bot Started")
        
        # Run initial scan
        self.scan_markets()
        
        # Schedule periodic scans
        schedule.every(1).hours.do(self.scan_markets)
        
        # Keep the bot running with a try-except block
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Bot runtime error: {e}")
        finally:
            self.loop.close()

# Pydantic Model for Request
class ScanRequest(BaseModel):
    pairs: list[str] = None

# FastAPI App
app = FastAPI(title="Forex Trading Signal Bot")
trading_bot = ForexLiveTradeBot()

@app.post("/scan-markets")
async def scan_markets(background_tasks: BackgroundTasks, request: ScanRequest = None):
    """
    Manually trigger market scan with improved concurrency handling
    """
    if request and request.pairs:
        specific_pairs = request.pairs
    else:
        specific_pairs = None
    
    # Use asyncio.to_thread for non-blocking background task
    background_tasks.add_task(asyncio.to_thread, trading_bot.scan_markets, specific_pairs)
    return {"status": "Scan initiated"}

@app.get("/health")
async def health_check():
    """
    Basic health check endpoint
    """
    return {
        "status": "alive",
        "current_pairs": trading_bot.pairs,
        "risk_amount": trading_bot.risk_amount,
        "min_risk_reward": trading_bot.min_risk_reward,
        "telegram_configured": bool(trading_bot.telegram_bot)
    }

def start_bot():
    """
    Start the trading bot in a separate thread with improved thread safety
    """
    bot_thread = threading.Thread(target=trading_bot.run, daemon=True)
    bot_thread.start()

def main():
    # Apply nest_asyncio again to ensure maximum compatibility
    nest_asyncio.apply()
    
    # Start bot thread
    start_bot()
    
    # Run FastAPI
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()