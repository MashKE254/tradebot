import os
import logging
import asyncio
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv

from fastapi import FastAPI
import uvicorn

from oandapyV20 import API
from oandapyV20.endpoints import instruments
from telegram import Bot

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OANDA_API_KEY = os.getenv("OANDA_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

class ForexSignalBot:
    def __init__(self, api_key: str, telegram_token: str, telegram_chat_id: str, pairs: List[str] = None):
        self.client = API(access_token=api_key)
        self.telegram_bot = Bot(token=telegram_token)
        self.chat_id = telegram_chat_id
        self.pairs = pairs or ['EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'GBP_USD', 'USD_CHF', 'GBP_JPY', 'XAU_USD']
        self.trading_sessions = {
            'EUR_USD': {
                'London': ('07:00', '16:00'),
                'New_York': ('12:00', '21:00')
            },
            'USD_JPY': {
                'Tokyo': ('23:00', '08:00'),
                'New_York': ('12:00', '21:00')
            },
            'AUD_USD': {
                'Sydney': ('21:00', '06:00'),
                'Tokyo': ('23:00', '08:00')
            },
        }

    async def send_telegram_message(self, message: str) -> None:
        """Send message through Telegram"""
        try:
            await self.telegram_bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            logger.info("Telegram message sent successfully")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def is_trading_session(self, pair: str, timestamp: datetime) -> bool:
        """Check if current time is within trading session for the pair"""
        try:
            if timestamp.tzinfo is None:
                timestamp = pytz.utc.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(pytz.UTC)
            
            hour = timestamp.hour
            minute = timestamp.minute
            current_time = hour + minute/60.0
            
            sessions = self.trading_sessions.get(pair, {})
            for session, (start, end) in sessions.items():
                start_hour, start_minute = map(int, start.split(':'))
                end_hour, end_minute = map(int, end.split(':'))
                
                session_start = start_hour + start_minute/60.0
                session_end = end_hour + end_minute/60.0
                
                if session_end < session_start:
                    if current_time >= session_start or current_time <= session_end:
                        return True
                else:
                    if session_start <= current_time <= session_end:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trading session: {str(e)}")
            return False

    def get_current_data(self, pair: str, count: int = 100, granularity: str = "H1") -> pd.DataFrame:
        """Fetch recent price data from Oanda"""
        try:
            params = {
                "count": count,
                "granularity": granularity
            }
            
            request = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = self.client.request(request)
            
            prices = []
            for candle in response['candles']:
                if candle['complete']:
                    prices.append({
                        'time': pd.to_datetime(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c'])
                    })
            
            df = pd.DataFrame(prices)
            if not df.empty:
                df.set_index('time', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            df = data.copy()
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
            
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return data

    def find_swing_points(self, data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Identify swing high and low points"""
        try:
            df = data.copy()
            df['SwingHigh'] = np.nan
            df['SwingLow'] = np.nan
            
            for i in range(lookback, len(df)-lookback):
                if (df['high'].iloc[i] > df['high'].iloc[i-lookback:i]).all() and \
                   (df['high'].iloc[i] > df['high'].iloc[i+1:i+lookback]).all():
                    df.iloc[i, df.columns.get_loc('SwingHigh')] = df['high'].iloc[i]
                
                if (df['low'].iloc[i] < df['low'].iloc[i-lookback:i]).all() and \
                   (df['low'].iloc[i] < df['low'].iloc[i+1:i+lookback]).all():
                    df.iloc[i, df.columns.get_loc('SwingLow')] = df['low'].iloc[i]
            
            return df
        
        except Exception as e:
            logger.error(f"Error finding swing points: {str(e)}")
            return data

    def format_signal_message(self, pair: str, signal_type: str, entry_price: float, 
                            stop_loss: float, take_profit: float) -> str:
        """Format the signal message for Telegram"""
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        
        message = f"""
🔔 <b>New {pair} Signal</b>

Type: {'🟢 BUY' if signal_type == 'LONG' else '🔴 SELL'}
Entry Price: {entry_price:.5f}
Stop Loss: {stop_loss:.5f}
Take Profit: {take_profit:.5f}
Risk/Reward: {risk_reward:.2f}

⚠️ <i>Always manage your risk and do your own analysis</i>
"""
        return message

    async def check_for_signals(self, pair: str) -> None:
        """Check for trading signals and send them through Telegram"""
        try:
            if not self.is_trading_session(pair, datetime.now(pytz.UTC)):
                logger.info(f"Pair {pair} is not in an active trading session.")
                return
            
            data = self.get_current_data(pair)
            if data.empty:
                logger.info(f"No data fetched for {pair}.")
                return
            
            data = self.calculate_indicators(data)
            data = self.find_swing_points(data)
            
            current_bar = data.iloc[-1]
            prev_bar = data.iloc[-2]
            
            if (current_bar['EMA50'] > current_bar['EMA200'] and
                current_bar['MACD_Hist'] > 0 and
                prev_bar['MACD_Hist'] <= 0 and
                current_bar['MACD'] < 0):
                
                recent_lows = data['SwingLow'].iloc[-20:].dropna()
                if not recent_lows.empty:
                    stop_loss = recent_lows.iloc[-1]
                    entry_price = current_bar['close']
                    risk = entry_price - stop_loss
                    take_profit = entry_price + (risk * 2.5)
                    
                    message = self.format_signal_message(pair, "LONG", entry_price, stop_loss, take_profit)
                    await self.send_telegram_message(message)
                    logger.info(f"Buy signal generated for {pair}")
            
            elif (current_bar['EMA50'] < current_bar['EMA200'] and
                  current_bar['MACD_Hist'] < 0 and
                  prev_bar['MACD_Hist'] >= 0 and
                  current_bar['MACD'] > 0):
                
                recent_highs = data['SwingHigh'].iloc[-20:].dropna()
                if not recent_highs.empty:
                    stop_loss = recent_highs.iloc[-1]
                    entry_price = current_bar['close']
                    risk = stop_loss - entry_price
                    take_profit = entry_price - (risk * 2.5)
                    
                    message = self.format_signal_message(pair, "SHORT", entry_price, stop_loss, take_profit)
                    await self.send_telegram_message(message)
                    logger.info(f"Sell signal generated for {pair}")
                    
        except Exception as e:
            logger.error(f"Error checking signals for {pair}: {str(e)}")

app = FastAPI()

forex_bot = ForexSignalBot(api_key=OANDA_API_KEY,
                           telegram_token=TELEGRAM_TOKEN,
                           telegram_chat_id=TELEGRAM_CHAT_ID)

@app.get("/")
async def root():
    return {"message": "Forex Signal Bot is running"}

@app.get("/check_signals")
async def check_signals(pair: str = None):
    """
    Endpoint to check for signals.
    If a specific pair is provided via query parameter, check that pair;
    otherwise, check all pairs concurrently.
    """
    if pair:
        await forex_bot.check_for_signals(pair)
        return {"message": f"Checked signals for {pair}"}
    else:
        tasks = [forex_bot.check_for_signals(p) for p in forex_bot.pairs]
        await asyncio.gather(*tasks)
        return {"message": "Checked signals for all pairs"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
