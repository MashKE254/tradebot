import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import pytz
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import Response
import uvicorn
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
from telegram import Bot

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexSignalBot:
    def __init__(self, api_key: str, telegram_token: str, telegram_chat_id: str,
                 pairs: List[str] = None, higher_timeframe: str = "H4", 
                 lower_timeframe: str = "M30"):
        self.client = API(access_token=api_key)
        self.telegram_bot = Bot(token=telegram_token)
        self.chat_id = telegram_chat_id
        self.pairs = pairs or ['EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 
                              'GBP_USD', 'GBP_JPY', 'XAU_USD', 'EUR_JPY', 
                              'AUD_JPY', 'USD_CHF', 'NZD_USD', 'EUR_GBP', 
                              'EUR_AUD', 'GBP_AUD', 'AUD_NZD', 'GBP_NZD']
        
        self.higher_timeframe = higher_timeframe
        self.lower_timeframe = lower_timeframe
        self.fixed_risk_amount = 100
        self.min_risk_reward = 3.0  # Corrected to backtest value
        
        # Risk management parameters
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.last_check_day = datetime.now(pytz.utc).date()

    async def send_telegram_message(self, message: str) -> None:
        """Send message through Telegram"""
        try:
            await self.telegram_bot.send_message(chat_id=self.chat_id, text=message, parse_mode='HTML')
            logger.info("Telegram message sent successfully")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def get_current_data(self, pair: str, granularity: str, count: int = 500) -> pd.DataFrame:
        """Fetch recent price data from Oanda"""
        try:
            params = {"count": count, "granularity": granularity}
            candles_list = []
            
            for r in InstrumentsCandlesFactory(instrument=pair, params=params):
                rv = self.client.request(r)
                candles_list.extend(rv["candles"])
            
            prices = []
            for candle in candles_list:
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
                df.sort_index(inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Calculate technical indicators with timeframe-optimized parameters"""
        try:
            df = data.copy()
            
            # Trend Indicators
            df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()

            # MACD Calculation
            fast_period = 8 if timeframe == "M30" else 12
            slow_period = 17 if timeframe == "M30" else 26
            signal_period = 6 if timeframe == "M30" else 9

            exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal']

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / avg_loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # ADX with +DI/-DI
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
            df['+DI'] = plus_di
            df['-DI'] = minus_di

            # ATR
            df['ATR'] = tr.rolling(14).mean()

            return df.ffill().dropna()

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

    def check_daily_loss_limit(self) -> bool:
        """Check if daily loss limit has been reached"""
        current_day = datetime.now(pytz.utc).date()
        if current_day != self.last_check_day:
            self.last_check_day = current_day
            self.daily_start_balance = self.current_balance
        
        daily_pnl = self.current_balance - self.daily_start_balance
        if daily_pnl <= -self.daily_start_balance * self.daily_loss_limit:
            logger.warning(f"Daily loss limit reached ({daily_pnl:.2f})")
            return True
        return False

    def format_signal_message(self, pair: str, signal_type: str, entry_price: float, 
                            stop_loss: float, take_profit: float) -> str:
        """Format the signal message for Telegram"""
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        message = f"""
🔔 <b>New {pair} Signal</b>

Type: {'🟢 BUY' if signal_type == 'LONG' else '🔴 SELL'}
Entry Price: {entry_price:.5f}
Stop Loss: {stop_loss:.5f}
Take Profit: {take_profit:.5f}
Risk/Reward: {risk_reward:.2f}
Position Size: {position_size:.2f} units

⚠️ <i>Fixed risk: ${self.fixed_risk_amount} per trade</i>
"""
        return message

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """Calculate position size based on fixed risk amount"""
        risk_per_unit = abs(entry_price - stop_loss)
        return self.fixed_risk_amount / risk_per_unit if risk_per_unit != 0 else 0

    async def check_for_signals(self, pair: str) -> None:
        """Check for trading signals using the backtest strategy"""
        try:
            if self.check_daily_loss_limit():
                return

            # Get data for both timeframes
            data_lower = self.get_current_data(pair, self.lower_timeframe)
            data_higher = self.get_current_data(pair, self.higher_timeframe)
            
            if data_lower.empty or data_higher.empty:
                return

            # Calculate indicators
            data_lower = self.calculate_indicators(data_lower, self.lower_timeframe)
            data_higher = self.calculate_indicators(data_higher, self.higher_timeframe)
            
            # Merge higher timeframe data
            h_tf_ema50 = f"{self.higher_timeframe}_EMA50"
            h_tf_ema200 = f"{self.higher_timeframe}_EMA200"
            data_lower[h_tf_ema50] = np.nan
            data_lower[h_tf_ema200] = np.nan
            
            for idx in data_lower.index:
                matching_higher_idx = data_higher.index.asof(idx)
                if not pd.isna(matching_higher_idx):
                    data_lower.at[idx, h_tf_ema50] = data_higher.at[matching_higher_idx, 'EMA50']
                    data_lower.at[idx, h_tf_ema200] = data_higher.at[matching_higher_idx, 'EMA200']
            
            data = data_lower.ffill().dropna()
            data = self.find_swing_points(data)
            
            current_bar = data.iloc[-1]
            prev_bar = data.iloc[-2]

            # Long Signal Conditions
            if (current_bar['EMA50'] > current_bar['EMA200'] and
                current_bar[h_tf_ema50] > current_bar[h_tf_ema200] and
                current_bar['MACD_Hist'] > 0 and
                prev_bar['MACD_Hist'] <= 0 and
                current_bar['MACD'] < 0 and
                current_bar['RSI'] < 65 and
                current_bar['ADX'] > 25):

                recent_lows = data['SwingLow'].iloc[-20:].dropna()
                if recent_lows.empty:
                    return
                
                stop_loss = recent_lows.iloc[-1]
                entry_price = current_bar['close']
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 3)  # 3:1 RR
                
                message = self.format_signal_message(pair, "LONG", entry_price, stop_loss, take_profit)
                await self.send_telegram_message(message)
                self.current_balance -= self.fixed_risk_amount  # Simulate risk

            # Short Signal Conditions
            elif (current_bar['EMA50'] < current_bar['EMA200'] and
                  current_bar[h_tf_ema50] < current_bar[h_tf_ema200] and
                  current_bar['MACD_Hist'] < 0 and
                  prev_bar['MACD_Hist'] >= 0 and
                  current_bar['MACD'] > 0 and
                  current_bar['RSI'] > 35 and
                  current_bar['ADX'] > 25):

                recent_highs = data['SwingHigh'].iloc[-20:].dropna()
                if recent_highs.empty:
                    return
                
                stop_loss = recent_highs.iloc[-1]
                entry_price = current_bar['close']
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 3)  # 3:1 RR
                
                message = self.format_signal_message(pair, "SHORT", entry_price, stop_loss, take_profit)
                await self.send_telegram_message(message)
                self.current_balance -= self.fixed_risk_amount  # Simulate risk

        except Exception as e:
            logger.error(f"Error checking signals for {pair}: {str(e)}")

app = FastAPI()
forex_bot = ForexSignalBot(
    api_key=os.getenv("OANDA_API_KEY"),
    telegram_token=os.getenv("TELEGRAM_TOKEN"),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID")
)

@app.get("/check_signals")
async def check_signals(pair: str = None):
    """Endpoint to check for signals"""
    if pair:
        await forex_bot.check_for_signals(pair)
        return {"message": f"Checked signals for {pair}"}
    else:
        tasks = [forex_bot.check_for_signals(p) for p in forex_bot.pairs]
        await asyncio.gather(*tasks)
        return {"message": "Checked signals for all pairs"}

@app.on_event("startup")
async def startup_event():
    startup_message = "Forex Signal Bot (Aligned with Backtest) is now running!"
    await forex_bot.send_telegram_message(startup_message)

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)