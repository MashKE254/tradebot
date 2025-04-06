import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import pytz
from datetime import datetime, timedelta
import logging
import asyncio
import telegram
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
import secrets
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import json
from dotenv import load_dotenv

# OANDA API imports
from oandapyV20 import API
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Forex Trading Signal Bot", 
              description="A bot that analyzes forex markets and sends trading signals via Telegram",
              version="1.0.0")

# Security for API endpoints
security = HTTPBasic()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("forex_bot.log")
    ]
)
logger = logging.getLogger("forex_bot")

# Scheduler for periodic tasks
scheduler = AsyncIOScheduler()

# Pydantic models for API requests and responses
class TelegramConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to enable Telegram notifications")
    chat_id: Optional[str] = Field(default=None, description="Telegram chat ID to send messages to")

class ForexBotConfig(BaseModel):
    pairs: List[str] = Field(
        default=['EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'GBP_USD', 'GBP_JPY', 
                 'XAU_USD', 'EUR_JPY', 'AUD_JPY', 'USD_CHF', 'NZD_USD', 'EUR_GBP', 
                 'EUR_AUD', 'GBP_AUD', 'AUD_NZD', 'GBP_NZD'],
        description="List of forex pairs to monitor"
    )
    timeframe: str = Field(default='M30', description="Timeframe for analysis")
    risk_amount: float = Field(default=100.0, description="Risk amount per trade")
    min_risk_reward: float = Field(default=1.75, description="Minimum risk-reward ratio")
    telegram: TelegramConfig = Field(default=TelegramConfig(), description="Telegram notification settings")

class TradeSignal(BaseModel):
    pair: str
    type: str  # LONG or SHORT
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float

class OANDAForexBot:
    def __init__(self, 
                 api_key: str = None,
                 pairs: List[str] = None, 
                 risk_amount: float = 100.0, 
                 min_risk_reward: float = 1.75,
                 telegram_bot_token: str = None,
                 telegram_chat_id: str = None,
                 environment: str = "practice"):
        """
        Initialize the Forex Trading Bot
        """
        self.api_key = api_key or os.getenv("OANDA_API_KEY")
        if not self.api_key:
            raise ValueError("OANDA API key is required")
        
        self.telegram_bot_token = telegram_bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        # Initialize OANDA API client
        self.client = API(
            access_token=self.api_key, 
            environment=environment,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        # Forex pairs to monitor
        self.pairs = pairs or ['EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 
                               'GBP_USD', 'GBP_JPY', 'XAU_USD', 'EUR_JPY', 'AUD_JPY', 
                               'USD_CHF', 'NZD_USD', 'EUR_GBP', 'EUR_AUD', 'GBP_AUD', 'AUD_NZD', 'GBP_NZD']
        self.risk_amount = risk_amount
        self.min_risk_reward = min_risk_reward
        
        # Tracking metrics
        self.active_signals = []
        self.historical_signals = []
        
        # Initialize Telegram bot if credentials provided
        self.telegram_bot = None
        if self.telegram_bot_token:
            self.telegram_bot = telegram.Bot(token=self.telegram_bot_token)
            logger.info("Telegram bot initialized")
        else:
            logger.warning("Telegram bot token not provided. Notifications disabled.")
    
    async def send_telegram_message(self, message: str):
        """
        Send a message via Telegram
        """
        if not self.telegram_bot or not self.telegram_chat_id:
            logger.warning("Telegram notification not sent: missing bot token or chat ID")
            return
        
        try:
            # Use MarkdownV2 instead of Markdown for better compatibility
            # Escape special characters to prevent parsing errors
            escaped_message = message.replace('.', '\\.').replace('-', '\\-').replace('+', '\\+').replace('(', '\\(').replace(')', '\\)').replace('!', '\\!')
            
            # Keep asterisks for bold formatting but ensure they're properly paired
            
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=escaped_message,
                parse_mode="MarkdownV2"  # Use MarkdownV2 instead of Markdown
            )
            logger.info(f"Telegram message sent: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {str(e)}")
            
            # Fallback: Try without markdown if parsing failed
            try:
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message.replace('*', ''),  # Remove markdown formatting
                    parse_mode=None  # No parsing
                )
                logger.info("Telegram message sent without markdown formatting")
            except Exception as e2:
                logger.error(f"Failed to send Telegram fallback message: {str(e2)}")
    
    async def load_historical_data(self, 
                                   pair: str, 
                                   timeframe: str = 'M30',
                                   count: int = 500) -> pd.DataFrame:
        """
        Load historical price data from OANDA API
        """
        try:
            logger.info(f"Fetching historical data for {pair}")
            
            # Set up end date (current time)
            end_date = datetime.utcnow()
            
            # Prepare API request parameters
            params = {
                "count": count,
                "granularity": timeframe,
                "price": "M"  # Midpoint pricing
            }
            
            # Create the instruments candles request
            r = instruments.InstrumentsCandles(
                instrument=pair,
                params=params
            )
            
            # Execute the request
            response = self.client.request(r)
            
            # Process the candles
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
            
            # Convert to DataFrame
            df = pd.DataFrame(prices)
            
            # Add pair information to index
            if not df.empty:
                df.set_index('time', inplace=True)
                df['pair'] = pair
            
            logger.info(f"Retrieved {len(df)} candles for {pair}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        :param data: OHLC price data
        :return: DataFrame with added indicator columns
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
    
    def find_trading_signals(self, data: pd.DataFrame) -> List[Dict]:
        """
        Find potential trading signals based on strategy rules
        
        :param data: DataFrame with price and indicator data
        :return: List of potential trade signals
        """
        signals = []
        
        # Ensure the pair is stored in the DataFrame
        pair = data['pair'].iloc[0] if not data.empty and 'pair' in data.columns else 'UNKNOWN'
        
        # Only check the most recent completed candle
        if len(data) < 2:  # Need at least 2 candles (previous + current)
            return signals
        
        current = data.iloc[-2]  # Previous completed candle
        next_bar = data.iloc[-1]  # Most recent candle
        
        # Long Entry Conditions
        long_conditions = (
            current['EMA50'] > current['EMA200'] and
            current['MACD_Hist'] <= 0 and
            next_bar['MACD_Hist'] > 0 and
            current['RSI'] < 65 and
            current['ADX'] > 25
        )
        
        # Short Entry Conditions
        short_conditions = (
            current['EMA50'] < current['EMA200'] and
            current['MACD_Hist'] >= 0 and
            next_bar['MACD_Hist'] < 0 and
            current['RSI'] > 35 and
            current['ADX'] > 25
        )
        
        # Generate lookback window for stop loss calculation
        lookback_window = 10
        pre_entry_data = data.iloc[-lookback_window-1:-1]  # Last 10 bars before current
        
        # Process Long signal
        if long_conditions:
            entry_price = next_bar['open']
            stop_loss = pre_entry_data['low'].min()
            take_profit = entry_price + (entry_price - stop_loss) * self.min_risk_reward
            
            signals.append({
                'type': 'LONG',
                'entry_price': entry_price,
                'entry_time': data.index[-1],
                'pair': pair,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': self.min_risk_reward
            })
        
        # Process Short signal
        if short_conditions:
            entry_price = next_bar['open']
            stop_loss = pre_entry_data['high'].max()
            take_profit = entry_price - (stop_loss - entry_price) * self.min_risk_reward
            
            signals.append({
                'type': 'SHORT',
                'entry_price': entry_price,
                'entry_time': data.index[-1],
                'pair': pair,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_reward_ratio': self.min_risk_reward
            })
        
        return signals
    
    async def scan_market(self, timeframe: str = 'M30'):
        """
        Scan the market for all currency pairs and generate signals
        """
        all_signals = []
        
        for pair in self.pairs:
            try:
                # Load historical data
                data = await self.load_historical_data(pair, timeframe=timeframe)
                
                # Skip if no data
                if data.empty:
                    continue
                
                # Calculate indicators
                indicators_data = self.calculate_indicators(data)
                
                # Find potential trade signals
                signals = self.find_trading_signals(indicators_data)
                
                # Process and notify for each signal
                for signal in signals:
                    all_signals.append(signal)
                    
                    # Create Telegram message for signal
                    message = f"*🚨 NEW {signal['type']} SIGNAL*\n\n" + \
                              f"*Pair:* {signal['pair']}\n" + \
                              f"*Entry Price:* {signal['entry_price']:.5f}\n" + \
                              f"*Stop Loss:* {signal['stop_loss']:.5f}\n" + \
                              f"*Take Profit:* {signal['take_profit']:.5f}\n" + \
                              f"*Risk/Reward:* {signal['risk_reward_ratio']:.2f}\n" + \
                              f"*Time:* {signal['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}\n\n" + \
                              f"📊 *Technical Analysis:*\n" + \
                              f"• EMA50 {'above' if indicators_data.iloc[-1]['EMA50'] > indicators_data.iloc[-1]['EMA200'] else 'below'} EMA200\n" + \
                              f"• MACD {'bullish crossover' if signal['type'] == 'LONG' else 'bearish crossover'}\n" + \
                              f"• RSI: {indicators_data.iloc[-1]['RSI']:.1f}\n" + \
                              f"• ADX: {indicators_data.iloc[-1]['ADX']:.1f}"
                    
                    # Send Telegram notification
                    await self.send_telegram_message(message)
                
                logger.info(f"Analyzed {pair}: {len(signals)} signals generated")
                
            except Exception as e:
                logger.error(f"Error analyzing {pair}: {str(e)}")
        
        return all_signals
    
    async def generate_daily_report(self):
        """
        Generate and send a daily market report
        """
        try:
            # Build report message
            report = f"*📈 FOREX DAILY REPORT - {datetime.now().strftime('%Y-%m-%d')}*\n\n"
            
            # Add report content
            report += f"*Active Trade Signals:* {len(self.active_signals)}\n"
            report += f"*Generated Today:* {len([s for s in self.historical_signals if s['entry_time'].date() == datetime.now().date()])}\n\n"
            
            # Market overview
            report += "*🌎 MARKET OVERVIEW:*\n"
            
            # Fetch latest price for major pairs
            for pair in ['EUR_USD', 'GBP_USD', 'USD_JPY', 'AUD_USD']:
                try:
                    data = await self.load_historical_data(pair, timeframe='H4', count=2)
                    if not data.empty:
                        price = data.iloc[-1]['close']
                        change = ((price / data.iloc[0]['close']) - 1) * 100
                        direction = "▲" if change > 0 else "▼"
                        report += f"• *{pair}:* {price:.5f} {direction} ({change:.2f}%)\n"
                except Exception as e:
                    logger.error(f"Error getting {pair} data for report: {str(e)}")
            
            # Send report
            await self.send_telegram_message(report)
            logger.info("Daily report generated and sent")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {str(e)}")

# Global variables
forex_bot = None
config = ForexBotConfig()

# Function to validate credentials
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = os.getenv("API_USERNAME", "admin")
    correct_password = os.getenv("API_PASSWORD", "password")
    
    is_username_correct = secrets.compare_digest(credentials.username, correct_username)
    is_password_correct = secrets.compare_digest(credentials.password, correct_password)
    
    if not (is_username_correct and is_password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# FastAPI startup event
@app.on_event("startup")
async def startup_event():
    global forex_bot
    
    # Initialize the forex bot
    forex_bot = OANDAForexBot(
        api_key=os.getenv("OANDA_API_KEY"),
        telegram_bot_token=os.getenv("TELEGRAM_TOKEN"),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
        pairs=config.pairs,
        risk_amount=config.risk_amount,
        min_risk_reward=config.min_risk_reward
    )
    
    # Schedule market scanning tasks every 5 minutes
    scheduler.add_job(
        forex_bot.scan_market, 
        'cron', 
        minute='*/5',  # Run every 5 minutes
        kwargs={"timeframe": config.timeframe}
    )
    
    # Schedule daily report at 00:05 daily
    scheduler.add_job(
        forex_bot.generate_daily_report,
        'cron',
        hour='0',
        minute='5'
    )
    
    # Start the scheduler
    scheduler.start()
    logger.info("Forex Trading Bot initialized and scheduled tasks started")
    
    # Send startup notification
    await forex_bot.send_telegram_message(
        "*🤖 FOREX TRADING BOT STARTED*\n\n" +
        f"Monitoring {len(config.pairs)} currency pairs\n" +
        f"Timeframe: {config.timeframe}\n" +
        "Bot will scan the market every 5 minutes and send signals when found."
    )

# FastAPI shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    # Stop the scheduler
    scheduler.shutdown()
    logger.info("Forex Trading Bot shutdown")

# API routes
@app.get("/")
async def root():
    return {"status": "online", "bot": "Forex Trading Signal Bot"}

# Manually scan market and return signals
@app.post("/scan", dependencies=[Depends(authenticate)])
async def scan_market(background_tasks: BackgroundTasks):
    """
    Manually trigger a market scan for trading signals
    """
    # Run scan in background to not block the response
    background_tasks.add_task(forex_bot.scan_market, config.timeframe)
    return {"message": "Market scan initiated. Check Telegram for signals."}

# Update bot configuration
@app.post("/config", dependencies=[Depends(authenticate)])
async def update_config(new_config: ForexBotConfig):
    """
    Update bot configuration
    """
    global config, forex_bot
    
    # Update global config
    config = new_config
    
    # Update forex bot instance
    if forex_bot:
        forex_bot.pairs = config.pairs
        forex_bot.risk_amount = config.risk_amount
        forex_bot.min_risk_reward = config.min_risk_reward
        
        # Update Telegram settings if provided
        if config.telegram.chat_id:
            forex_bot.telegram_chat_id = config.telegram.chat_id
    
    # Restart scheduler with new settings
    scheduler.remove_all_jobs()
    
    # Add scanning job every 5 minutes
    scheduler.add_job(
        forex_bot.scan_market, 
        'cron', 
        minute='*/5',  # Run every 5 minutes
        kwargs={"timeframe": config.timeframe}
    )
    
    # Add daily report job
    scheduler.add_job(
        forex_bot.generate_daily_report,
        'cron',
        hour='0',
        minute='5'
    )
    
    return {"message": "Configuration updated successfully", "config": config}

# Get active signals
@app.get("/signals", dependencies=[Depends(authenticate)])
async def get_signals():
    """
    Get all active trading signals
    """
    if forex_bot:
        return {"signals": forex_bot.active_signals}
    return {"signals": []}

# Get bot status and information
@app.get("/status", dependencies=[Depends(authenticate)])
async def get_status():
    """
    Get bot status information
    """
    if not forex_bot:
        return {"status": "not_initialized"}
    
    # Construct status response
    status_info = {
        "status": "running",
        "pairs_monitored": len(forex_bot.pairs),
        "timeframe": config.timeframe,
        "active_signals": len(forex_bot.active_signals),
        "historical_signals": len(forex_bot.historical_signals),
        "telegram_enabled": bool(forex_bot.telegram_bot and forex_bot.telegram_chat_id),
        "uptime": "N/A"  # Could add uptime tracking
    }
    
    return status_info

# Send test notification
@app.post("/test-notification", dependencies=[Depends(authenticate)])
async def test_notification():
    """
    Send a test notification to Telegram
    """
    if not forex_bot:
        raise HTTPException(status_code=400, detail="Bot not initialized")
    
    await forex_bot.send_telegram_message(
        "*🧪 TEST NOTIFICATION*\n\n" +
        "This is a test message to verify that your Telegram notifications are working correctly."
    )
    
    return {"message": "Test notification sent"}

# Main entry point
if __name__ == "__main__":
    # Get port from environment variable for Railway compatibility
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI application
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)