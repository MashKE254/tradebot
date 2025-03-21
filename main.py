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
import oandapyV20.endpoints.instruments as instruments
from telegram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from functools import partial

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
        
        # Update the API initialization with environment and proper headers
        logger.info(f"setting up API-client for environment practice")
        logger.info(f"applying headers Authorization")
        self.client = API(
            access_token=api_key, 
            environment="practice",  # Explicitly set to practice or live
            headers={"Authorization": f"Bearer {api_key}"})
        
        self.telegram_bot = Bot(token=telegram_token)
        self.chat_id = telegram_chat_id
        self.pairs = pairs or ['EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 
                              'GBP_USD', 'GBP_JPY', 'XAU_USD', 'EUR_JPY', 
                              'AUD_JPY', 'USD_CHF', 'NZD_USD', 'EUR_GBP', 
                              'EUR_AUD', 'GBP_AUD', 'AUD_NZD', 'GBP_NZD']
        
        self.higher_timeframe = higher_timeframe
        self.lower_timeframe = lower_timeframe
        self.fixed_risk_amount = 100
        self.min_risk_reward = 1.5
        
        # Risk management
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.daily_loss_limit = 0.05
        self.last_check_day = datetime.now(pytz.utc).date()
        
        # Initialize API connection checks and attributes
        self._check_api_connection()
        
    def _check_api_connection(self):
        """Verify API connectivity by fetching test data"""
        try:
            logger.info("Checking API connection...")
            # Test connection by making a direct request instead of using get_current_data
            
            params = {
                "count": 1,
                "granularity": "M1"
            }
            
            request = instruments.InstrumentsCandles(
                instrument="EUR_USD",
                params=params
            )
            
            # The issue is here - InstrumentsCandles doesn't have an endpoint attribute
            # Instead we should just log that we're making a request
            logger.info(f"performing request for EUR_USD candles")
            response = self.client.request(request)
            
            # Check if the response contains candles data
            if "candles" in response and len(response["candles"]) > 0:
                logger.info("API connection successful")
            else:
                raise ConnectionError("API returned empty candles data")
                
        except Exception as e:
            logger.error(f"API connection failed: {str(e)}")
            raise RuntimeError("Failed to connect to OANDA API") from e
        
        # Candle alignment tracking
        self.last_checked = {}
        self.scheduled_checks = asyncio.Lock()
        self.active_positions = {}

    async def send_telegram_message(self, message: str) -> None:
        try:
            await self.telegram_bot.send_message(
                chat_id=self.chat_id, 
                text=message, 
                parse_mode='HTML'
            )
            logger.info("Telegram message sent")
        except Exception as e:
            logger.error(f"Telegram error: {str(e)}")

    def get_current_data(self, pair: str, granularity: str, count: int = 500) -> pd.DataFrame:
        try:
            logger.info(f"Fetching {count} {granularity} candles for {pair}")
            params = {"count": count, "granularity": granularity}
            candles_list = []
            
            for r in InstrumentsCandlesFactory(instrument=pair, params=params):
                logger.info(f"performing request for {pair} candles")
                rv = self.client.request(r)
                candles_list.extend(rv["candles"])
            
            logger.info(f"Received {len(candles_list)} candles for {pair}")
            
            if not candles_list:
                logger.warning(f"No candles received for {pair}")
                return pd.DataFrame()
            
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
            
            logger.info(f"Processed {len(prices)} complete candles for {pair}")
            
            if not prices:
                logger.warning(f"No complete candles for {pair}")
                return pd.DataFrame()
            
            df = pd.DataFrame(prices)
            if not df.empty:
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                
            return df
            
        except Exception as e:
            logger.error(f"Data error {pair}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
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
            logger.error(f"Indicator error: {str(e)}")
            return data

    def find_swing_points(self, data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
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
            logger.error(f"Swing point error: {str(e)}")
            return data

    def check_daily_loss_limit(self) -> bool:
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
        risk_reward = abs(take_profit - entry_price) / abs(stop_loss - entry_price)
        position_size = self.calculate_position_size(entry_price, stop_loss)
        
        return f"""
🔔 <b>New {pair} Signal</b>

Type: {'🟢 BUY' if signal_type == 'LONG' else '🔴 SELL'}
Entry Price: {entry_price:.5f}
Stop Loss: {stop_loss:.5f}
Take Profit: {take_profit:.5f}
Risk/Reward: {risk_reward:.2f}
Position Size: {position_size:.2f} units

⚠️ <i>Fixed risk: ${self.fixed_risk_amount} per trade</i>
"""

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        risk_per_unit = abs(entry_price - stop_loss)
        return self.fixed_risk_amount / risk_per_unit if risk_per_unit != 0 else 0

    async def check_for_signals(self, pair: str) -> None:
        try:
            # Rate limiting
            async with self.scheduled_checks:
                now = datetime.now(pytz.utc)
                if pair in self.last_checked:
                    elapsed = now - self.last_checked[pair]
                    if elapsed < timedelta(minutes=5):
                        return
                
                
                self.last_checked[pair] = now

                if self.check_daily_loss_limit():
                    return

                data_lower = self.get_current_data(pair, self.lower_timeframe)
                data_higher = self.get_current_data(pair, self.higher_timeframe)
                
                if data_lower.empty or data_higher.empty:
                    return

                data_lower = self.calculate_indicators(data_lower, self.lower_timeframe)
                data_higher = self.calculate_indicators(data_higher, self.higher_timeframe)
                
                # Add validation
                if data_higher['EMA200'].isna().any():
                    logger.error(f"Insufficient H4 data for {pair}")
                    return
                
                h_tf_ema50 = f"{self.higher_timeframe}_EMA50"
                h_tf_ema200 = f"{self.higher_timeframe}_EMA200"
  
  
                
                for idx in data_lower.index:
                    matching_higher_idx = data_higher.index.asof(idx)
                    if not pd.isna(matching_higher_idx):
                        data_lower.at[idx, h_tf_ema50] = data_higher.at[matching_higher_idx, 'EMA50']
                        data_lower.at[idx, h_tf_ema200] = data_higher.at[matching_higher_idx, 'EMA200']
                
                data = data_lower.ffill().dropna()
                data = self.find_swing_points(data)
                
                current_bar = data.iloc[-1]
                prev_bar = data.iloc[-2]
                
                debug_info = f"""
        {pair} Signal Check:
        EMA50 > EMA200: {current_bar['EMA50'] > current_bar['EMA200']}
        HTF EMA50 > HTF EMA200: {current_bar[h_tf_ema50] > current_bar[h_tf_ema200]}
        MACD Hist > 0: {current_bar['MACD_Hist'] > 0}
        ADX > 25: {current_bar['ADX'] > 25}
        RSI: {current_bar['RSI']}
        """
                logger.debug(debug_info)

                # Long Signal
                if (current_bar['EMA50'] > current_bar['EMA200'] and
                    current_bar[h_tf_ema50] > current_bar[h_tf_ema200] and
                    current_bar['MACD_Hist'] > 0 and
                    prev_bar['MACD_Hist'] <= 0 and
                    current_bar['MACD'] < 0 and
                    current_bar['RSI'] < 65 and
                    current_bar['ADX'] > 25):

                    recent_lows = data['SwingLow'].iloc[-20:].dropna()
                    if not recent_lows.empty:
                        stop_loss = recent_lows.iloc[-1]
                        entry_price = current_bar['close']
                        risk = entry_price - stop_loss
                        take_profit = entry_price + (risk * self.min_risk_reward)
                        
                        if take_profit > entry_price:
                            message = self.format_signal_message(pair, "LONG", entry_price, stop_loss, take_profit)
                            await self.send_telegram_message(message)
                            
                            # Track position
                            self.active_positions[pair] = {
                                'type': 'LONG',
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'size': self.calculate_position_size(entry_price, stop_loss)
                            }

                # Short Signal
                elif (current_bar['EMA50'] < current_bar['EMA200'] and
                      current_bar[h_tf_ema50] < current_bar[h_tf_ema200] and
                      current_bar['MACD_Hist'] < 0 and
                      prev_bar['MACD_Hist'] >= 0 and
                      current_bar['MACD'] > 0 and
                      current_bar['RSI'] > 35 and
                      current_bar['ADX'] > 25):

                    recent_highs = data['SwingHigh'].iloc[-20:].dropna()
                    if not recent_highs.empty:
                        stop_loss = recent_highs.iloc[-1]
                        entry_price = current_bar['close']
                        risk = stop_loss - entry_price
                        take_profit = entry_price - (risk * self.min_risk_reward)
                        
                        if take_profit < entry_price:
                            message = self.format_signal_message(pair, "SHORT", entry_price, stop_loss, take_profit)
                            await self.send_telegram_message(message)
                            
                            self.active_positions[pair] = {
                                'type': 'SHORT',
                                'entry': entry_price,
                                'sl': stop_loss,
                                'tp': take_profit,
                                'size': self.calculate_position_size(entry_price, stop_loss)
                            }

                # Check existing positions
                if pair in self.active_positions:
                    position = self.active_positions[pair]
                    current_low = current_bar['low']
                    current_high = current_bar['high']
                    
                    if position['type'] == 'LONG':
                        if current_low <= position['sl']:
                            pl = (position['sl'] - position['entry']) * position['size']
                            self.current_balance += pl
                            del self.active_positions[pair]
                        elif current_high >= position['tp']:
                            pl = (position['tp'] - position['entry']) * position['size']
                            self.current_balance += pl
                            del self.active_positions[pair]
                    
                    else:  # SHORT
                        if current_high >= position['sl']:
                            pl = (position['entry'] - position['sl']) * position['size']
                            self.current_balance += pl
                            del self.active_positions[pair]
                        elif current_low <= position['tp']:
                            pl = (position['entry'] - position['tp']) * position['size']
                            self.current_balance += pl
                            del self.active_positions[pair]

        except Exception as e:
            logger.error(f"Signal check error {pair}: {str(e)}")
            
app = FastAPI()
forex_bot = ForexSignalBot(
    api_key=os.getenv("OANDA_API_KEY"),
    telegram_token=os.getenv("TELEGRAM_TOKEN"),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID")
)

# Create a single global scheduler
scheduler = AsyncIOScheduler(timezone="UTC")

# Create the async functions that will be directly called by the scheduler
async def check_all_signals_job():
    """Check all currency pairs"""
    tasks = [forex_bot.check_for_signals(pair) for pair in forex_bot.pairs]
    await asyncio.gather(*tasks)
    logger.info("Completed full signal check at %s", datetime.utcnow())

async def check_h4_confirmation_job():
    """Update higher timeframe trends"""
    logger.info("Updating H4 trend confirmations")
    # Add H4-specific logic here

@app.on_event("startup")
async def startup_event():
    # Configure scheduler with explicit job store and executor
    job_defaults = {
        'misfire_grace_time': 300,
        'coalesce': True,
        'max_instances': 1
    }
    
    scheduler = AsyncIOScheduler(
        job_defaults=job_defaults,
        timezone="UTC"
    )

    scheduler.add_job(
        check_all_signals_job,
        CronTrigger(minute="0,30", second="5"),
        name="30m_signals_check"
    )
    
    scheduler.add_job(
        check_h4_confirmation_job,
        CronTrigger(hour="0,4,8,12,16,20", minute="0", second="10"),
        name="4h_trend_update"
    )
    
    scheduler.start()
    logger.info("Scheduler started with jobs: %s", scheduler.get_jobs())

@app.on_event("shutdown")
async def shutdown_event():
    # Properly shut down the scheduler when the application stops
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler shutdown")

@app.get("/")
async def root():
    return {"status": "active", "balance": forex_bot.current_balance}

@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/check_signals")
async def manual_check(pair: str = None):
    """Manual trigger endpoint"""
    if pair:
        await forex_bot.check_for_signals(pair)
    else:
        await check_all_signals_job()
    return {"status": "checked"}


if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)