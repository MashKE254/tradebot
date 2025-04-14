import os
import asyncio
import logging
import pytz
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Set
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
import io
import base64
import telegram
from telegram.error import TelegramError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

# OANDA API imports
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
from oandapyV20.contrib.factories import InstrumentsCandlesFactory
import oandapyV20.endpoints.instruments as instruments

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("forex_bot.log")
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Forex Trading Signal Bot", description="Real-time forex trading signals based on technical indicators")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class TradeSignal(BaseModel):
    pair: str
    type: str
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    risk_reward: float

class BacktestResult(BaseModel):
    pair: str
    type: str
    entry_price: float
    entry_time: datetime
    exit_price: float
    exit_time: datetime
    result: str
    profit: float

class ForexTradingBot:
    def __init__(
        self,
        api_key: str = None,
        pairs: List[str] = None,
        initial_balance: float = 10000.0,
        risk_amount: float = 100.0,
        min_risk_reward: float = 1.75,
        daily_loss_limit: float = 0.05,
        environment: str = "practice",
        telegram_token: str = None,
        telegram_chat_id: str = None,
        timeframe: str = "H1"
    ):
        # API setup
        self.api_key = api_key or os.getenv("OANDA_API_KEY")
        if not self.api_key:
            raise ValueError("OANDA API key is required")
        
        self.client = API(
            access_token=self.api_key,
            environment=environment,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        
        # Trading parameters
        self.pairs = pairs or [
            'EUR_USD', 'USD_JPY', 'AUD_USD', 'USD_CAD', 'GBP_USD', 
            'GBP_JPY', 'XAU_USD', 'EUR_JPY', 'AUD_JPY', 'USD_CHF', 
            'NZD_USD', 'EUR_GBP', 'EUR_AUD', 'GBP_AUD', 'AUD_NZD', 'GBP_NZD'
        ]
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_amount = risk_amount
        self.min_risk_reward = min_risk_reward
        self.daily_loss_limit = daily_loss_limit
        self.timeframe = timeframe
        
        # Telegram setup
        self.telegram_token = telegram_token or os.getenv("TELEGRAM_TOKEN")
        self.telegram_chat_id = telegram_chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.telegram_bot = None
        if self.telegram_token and self.telegram_chat_id:
            self.telegram_bot = telegram.Bot(token=self.telegram_token)
        
        # State tracking
        self.active_trades: Dict[str, TradeSignal] = {}
        self.historical_trades: List[BacktestResult] = []
        self.last_check_time = datetime.utcnow() - timedelta(hours=1)  # Initialize to 1 hour ago
        
        # Create scheduler for periodic tasks
        self.scheduler = AsyncIOScheduler()
        
        logger.info(f"Forex Trading Bot initialized with {len(self.pairs)} currency pairs")

    async def send_telegram_message(self, message: str) -> bool:
        """Send message to Telegram chat"""
        if not self.telegram_bot or not self.telegram_chat_id:
            logger.warning("Telegram bot not configured. Message not sent.")
            return False
        
        try:
            await self.telegram_bot.send_message(
                chat_id=self.telegram_chat_id,
                text=message,
                parse_mode='Markdown'
            )
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def send_telegram_image(self, image_data, caption: str = None) -> bool:
        """Send image to Telegram chat"""
        if not self.telegram_bot or not self.telegram_chat_id:
            logger.warning("Telegram bot not configured. Image not sent.")
            return False
        
        try:
            await self.telegram_bot.send_photo(
                chat_id=self.telegram_chat_id,
                photo=image_data,
                caption=caption,
                parse_mode='Markdown'
            )
            return True
        except TelegramError as e:
            logger.error(f"Failed to send Telegram image: {e}")
            return False

    async def fetch_historical_data(
        self,
        pair: str,
        timeframe: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        count: int = 200
    ) -> pd.DataFrame:
        """Fetch historical price data from OANDA API"""
        try:
            logger.info(f"Fetching historical data for {pair}")
            
            # Use provided timeframe or default
            tf = timeframe or self.timeframe
            
            # Set default end_date to now if not provided
            if end_date is None:
                end_date = datetime.utcnow()
            
            # Set default start_date if not provided
            if start_date is None:
                if tf == 'H1':
                    # For H1, get 200 candles (about 8 days worth)
                    start_date = end_date - timedelta(days=8)
                elif tf == 'H4':
                    # For H4, get 200 candles (about 33 days worth)
                    start_date = end_date - timedelta(days=33)
                else:
                    # Default to 30 days
                    start_date = end_date - timedelta(days=30)
            
            # Prepare API request parameters
            params = {
                "from": start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "to": end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "granularity": tf,
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
        
        except V20Error as e:
            logger.error(f"OANDA API error for {pair}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {str(e)}")
            return pd.DataFrame()

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for strategy"""
        if data.empty:
            return data
        
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

    def check_entry_conditions(self, data: pd.DataFrame) -> Optional[TradeSignal]:
        """Check if current market conditions meet entry criteria"""
        if data.empty or len(data) < 2:
            return None
        
        # Get last two candles for entry condition check
        current = data.iloc[-2]
        next_bar = data.iloc[-1]
        pair = current['pair']
        
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
        
        if long_conditions:
            entry_price = next_bar['open']
            # Simulate finding stop loss and take profit
            lookback_window = 10
            pre_entry_data = data.iloc[-lookback_window-2:-2]
            stop_loss = pre_entry_data['low'].min()
            take_profit = entry_price + (entry_price - stop_loss) * self.min_risk_reward
            
            return TradeSignal(
                pair=pair,
                type="LONG",
                entry_price=entry_price,
                entry_time=data.index[-1],
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=self.min_risk_reward
            )
        
        if short_conditions:
            entry_price = next_bar['open']
            # Simulate finding stop loss and take profit
            lookback_window = 10
            pre_entry_data = data.iloc[-lookback_window-2:-2]
            stop_loss = pre_entry_data['high'].max()
            take_profit = entry_price - (stop_loss - entry_price) * self.min_risk_reward
            
            return TradeSignal(
                pair=pair,
                type="SHORT",
                entry_price=entry_price,
                entry_time=data.index[-1],
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward=self.min_risk_reward
            )
        
        return None

    def check_trade_exits(self, active_trade: TradeSignal, current_data: pd.DataFrame) -> Optional[BacktestResult]:
        """Check if an active trade should be closed"""
        if current_data.empty:
            return None
        
        # Get current price data
        current_bar = current_data.iloc[-1]
        current_time = current_data.index[-1]
        
        if active_trade.type == "LONG":
            # Check stop loss
            if current_bar['low'] <= active_trade.stop_loss:
                return BacktestResult(
                    pair=active_trade.pair,
                    type=active_trade.type,
                    entry_price=active_trade.entry_price,
                    entry_time=active_trade.entry_time,
                    exit_price=active_trade.stop_loss,
                    exit_time=current_time,
                    result="STOP_LOSS",
                    profit=(active_trade.stop_loss - active_trade.entry_price) * 
                           self.risk_amount / (active_trade.entry_price - active_trade.stop_loss)
                )
            
            # Check take profit
            if current_bar['high'] >= active_trade.take_profit:
                return BacktestResult(
                    pair=active_trade.pair,
                    type=active_trade.type,
                    entry_price=active_trade.entry_price,
                    entry_time=active_trade.entry_time,
                    exit_price=active_trade.take_profit,
                    exit_time=current_time,
                    result="TAKE_PROFIT",
                    profit=(active_trade.take_profit - active_trade.entry_price) * 
                           self.risk_amount / (active_trade.entry_price - active_trade.stop_loss)
                )
        
        elif active_trade.type == "SHORT":
            # Check stop loss
            if current_bar['high'] >= active_trade.stop_loss:
                return BacktestResult(
                    pair=active_trade.pair,
                    type=active_trade.type,
                    entry_price=active_trade.entry_price,
                    entry_time=active_trade.entry_time,
                    exit_price=active_trade.stop_loss,
                    exit_time=current_time,
                    result="STOP_LOSS",
                    profit=(active_trade.entry_price - active_trade.stop_loss) * 
                           self.risk_amount / (active_trade.stop_loss - active_trade.entry_price)
                )
            
            # Check take profit
            if current_bar['low'] <= active_trade.take_profit:
                return BacktestResult(
                    pair=active_trade.pair,
                    type=active_trade.type,
                    entry_price=active_trade.entry_price,
                    entry_time=active_trade.entry_time,
                    exit_price=active_trade.take_profit,
                    exit_time=current_time,
                    result="TAKE_PROFIT",
                    profit=(active_trade.entry_price - active_trade.take_profit) * 
                           self.risk_amount / (active_trade.stop_loss - active_trade.entry_price)
                )
        
        return None

    async def generate_chart(self, pair: str) -> io.BytesIO:
        """Generate chart for a specific pair"""
        data = await self.fetch_historical_data(pair, count=100)
        if data.empty:
            raise ValueError(f"No data available for {pair}")
        
        data = self.calculate_indicators(data)
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [4, 1, 1]})
        
        # Plot price and EMAs on top subplot
        ax1.plot(data.index, data['close'], label='Close Price')
        ax1.plot(data.index, data['EMA50'], label='EMA50', color='orange')
        ax1.plot(data.index, data['EMA200'], label='EMA200', color='red')
        
        # Add active trades if any
        if pair in self.active_trades:
            trade = self.active_trades[pair]
            entry_time = trade.entry_time
            ax1.axvline(x=entry_time, color='green' if trade.type == "LONG" else 'red', linestyle='--')
            ax1.axhline(y=trade.entry_price, color='blue', linestyle='-')
            ax1.axhline(y=trade.stop_loss, color='red', linestyle='-')
            ax1.axhline(y=trade.take_profit, color='green', linestyle='-')
        
        ax1.set_title(f'{pair} - {self.timeframe} Timeframe')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot MACD on middle subplot
        ax2.plot(data.index, data['MACD'], label='MACD', color='blue')
        ax2.plot(data.index, data['Signal'], label='Signal', color='red')
        ax2.bar(data.index, data['MACD_Hist'], label='Histogram', color='green', width=0.02, 
                alpha=0.5)
        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        # Plot RSI on bottom subplot
        ax3.plot(data.index, data['RSI'], label='RSI', color='purple')
        ax3.axhline(y=70, color='red', linestyle='--')
        ax3.axhline(y=30, color='green', linestyle='--')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        plt.tight_layout()
        
        # Save to BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        
        return buf

    async def scan_for_signals(self) -> List[TradeSignal]:
        """Scan all currency pairs for trading signals"""
        signals = []
        
        for pair in self.pairs:
            # Skip if we already have an active trade for this pair
            if pair in self.active_trades:
                continue
                
            try:
                # Fetch and prepare data
                data = await self.fetch_historical_data(pair)
                if data.empty:
                    logger.warning(f"No data available for {pair}, skipping...")
                    continue
                
                data = self.calculate_indicators(data)
                
                # Check for entry conditions
                signal = self.check_entry_conditions(data)
                if signal:
                    signals.append(signal)
                    logger.info(f"Found signal for {pair}: {signal.type} at {signal.entry_price}")
            except Exception as e:
                logger.error(f"Error scanning {pair}: {str(e)}")
        
        return signals

    async def update_active_trades(self) -> List[BacktestResult]:
        """Update status of active trades and check for exits"""
        closed_trades = []
        
        for pair, trade in list(self.active_trades.items()):
            try:
                # Fetch latest data
                data = await self.fetch_historical_data(pair)
                if data.empty:
                    logger.warning(f"No data available to update trade for {pair}")
                    continue
                
                # Check if trade should be closed
                result = self.check_trade_exits(trade, data)
                if result:
                    closed_trades.append(result)
                    self.historical_trades.append(result)
                    del self.active_trades[pair]
                    
                    # Log and send notification
                    logger.info(f"Closed trade for {pair}: {result.result} with profit {result.profit:.2f}")
                    
                    # Send notification to Telegram
                    message = (
                        f"🔔 *TRADE CLOSED*\n"
                        f"Pair: *{pair}*\n"
                        f"Type: *{result.type}*\n"
                        f"Result: *{result.result}*\n"
                        f"Entry: {result.entry_price:.5f}\n"
                        f"Exit: {result.exit_price:.5f}\n"
                        f"Profit: *${result.profit:.2f}*\n"
                        f"Time Open: {(result.exit_time - result.entry_time).total_seconds() / 3600:.1f} hours"
                    )
                    await self.send_telegram_message(message)
                    
                    # Generate and send chart
                    try:
                        chart_buf = await self.generate_chart(pair)
                        await self.send_telegram_image(chart_buf, f"Chart for {pair} after trade closed")
                    except Exception as e:
                        logger.error(f"Failed to generate or send chart: {e}")
            
            except Exception as e:
                logger.error(f"Error updating trade for {pair}: {str(e)}")
        
        return closed_trades

    async def run_scanner(self):
        """Main scanner function to run periodically"""
        logger.info("Running trading signal scanner")
        
        try:
            # Check for new signals
            signals = await self.scan_for_signals()
            
            # Process new signals
            for signal in signals:
                # Record the signal as an active trade
                self.active_trades[signal.pair] = signal
                
                # Send notification
                message = (
                    f"🔔 *NEW SIGNAL DETECTED*\n"
                    f"Pair: *{signal.pair}*\n"
                    f"Type: *{signal.type}*\n"
                    f"Entry Price: {signal.entry_price:.5f}\n"
                    f"Stop Loss: {signal.stop_loss:.5f}\n"
                    f"Take Profit: {signal.take_profit:.5f}\n"
                    f"Risk/Reward: {signal.risk_reward:.2f}\n"
                    f"Time: {signal.entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                )
                await self.send_telegram_message(message)
                
                # Generate and send chart
                try:
                    chart_buf = await self.generate_chart(signal.pair)
                    await self.send_telegram_image(chart_buf, f"Entry chart for {signal.pair}")
                except Exception as e:
                    logger.error(f"Failed to generate or send chart: {e}")
            
            # Update active trades
            await self.update_active_trades()
            
            # Update last check time
            self.last_check_time = datetime.utcnow()
            
            # Generate performance report if we have closed trades
            if self.historical_trades and len(self.historical_trades) % 5 == 0:  # Every 5 trades
                await self.send_performance_report()
                
        except Exception as e:
            logger.error(f"Error in scanner run: {str(e)}")

    async def send_performance_report(self):
        """Generate and send a performance report"""
        if not self.historical_trades:
            return
        
        try:
            # Calculate statistics
            total_trades = len(self.historical_trades)
            winning_trades = sum(1 for t in self.historical_trades if t.profit > 0)
            losing_trades = total_trades - winning_trades
            total_profit = sum(t.profit for t in self.historical_trades)
            win_rate = (winning_trades / total_trades * 100) if total_trades else 0
            
            # Generate report message
            report = (
                f"📊 *PERFORMANCE REPORT*\n"
                f"Total Trades: *{total_trades}*\n"
                f"Winning Trades: *{winning_trades}*\n"
                f"Losing Trades: *{losing_trades}*\n"
                f"Win Rate: *{win_rate:.2f}%*\n"
                f"Total Profit: *${total_profit:.2f}*\n\n"
                f"*Recent Trades:*\n"
            )
            
            # Add recent trades info
            for trade in sorted(self.historical_trades[-5:], key=lambda x: x.exit_time, reverse=True):
                report += (
                    f"• {trade.pair} {trade.type}: {trade.result} (${trade.profit:.2f})\n"
                )
            
            # Send report
            await self.send_telegram_message(report)
            
            # Generate equity curve
            if len(self.historical_trades) >= 10:
                plt.figure(figsize=(10, 6))
                cumulative_profit = np.cumsum([t.profit for t in sorted(self.historical_trades, key=lambda x: x.exit_time)])
                plt.plot(cumulative_profit)
                plt.title('Equity Curve')
                plt.xlabel('Trade Number')
                plt.ylabel('Cumulative Profit ($)')
                plt.grid(True)
                
                # Save to BytesIO
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                plt.close()
                
                # Send chart
                await self.send_telegram_image(buf, "Equity Curve")
                
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")

    async def start_scheduler(self):
        """Start the background scheduler"""
        # Add scanner job based on timeframe
        if self.timeframe == 'H1':
            self.scheduler.add_job(
                self.run_scanner,
                trigger=CronTrigger(minute=1),  # Run at 1 minute past every hour
                id='scanner_job',
                replace_existing=True
            )
        elif self.timeframe == 'H4':
            self.scheduler.add_job(
                self.run_scanner,
                trigger=CronTrigger(hour='*/4', minute=1),  # Run every 4 hours
                id='scanner_job',
                replace_existing=True
            )
        else:
            # Default to hourly
            self.scheduler.add_job(
                self.run_scanner,
                trigger=CronTrigger(minute=1),  # Run at 1 minute past every hour
                id='scanner_job',
                replace_existing=True
            )
        
        # Add daily report job
        self.scheduler.add_job(
            self.send_performance_report,
            trigger=CronTrigger(hour=0, minute=1),  # Run daily at 00:01
            id='daily_report',
            replace_existing=True
        )
        
        # Start the scheduler
        self.scheduler.start()
        logger.info(f"Scheduler started. Running scanner on {self.timeframe} timeframe.")

# Create global bot instance
bot = ForexTradingBot(
    api_key=os.getenv("OANDA_API_KEY"),
    telegram_token=os.getenv("TELEGRAM_TOKEN"),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID"),
    timeframe=os.getenv("TIMEFRAME", "H1")
)

# API routes
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Forex Trading Signal Bot is running",
        "status": "active",
        "timeframe": bot.timeframe,
        "active_trades": len(bot.active_trades),
        "historical_trades": len(bot.historical_trades)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/trades/active")
async def get_active_trades():
    """Get all active trades"""
    return [trade.dict() for trade in bot.active_trades.values()]

@app.get("/trades/history")
async def get_trade_history(limit: int = 20):
    """Get historical trades with optional limit"""
    return [trade.dict() for trade in sorted(
        bot.historical_trades, 
        key=lambda t: t.exit_time, 
        reverse=True
    )[:limit]]

@app.get("/pairs")
async def get_pairs():
    """Get all monitored currency pairs"""
    return {"pairs": bot.pairs}

@app.post("/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    """Manually trigger a scan for signals"""
    background_tasks.add_task(bot.run_scanner)
    return {"message": "Scan triggered"}

@app.post("/report")
async def trigger_report(background_tasks: BackgroundTasks):
    """Manually trigger a performance report"""
    background_tasks.add_task(bot.send_performance_report)
    return {"message": "Report generation triggered"}

@app.get("/charts/{pair}")
async def get_chart(pair: str):
    """Generate and return a chart for a specific pair"""
    if pair not in bot.pairs:
        raise HTTPException(status_code=404, detail=f"Pair {pair} not found")
    
    try:
        # Generate the chart
        chart_buf = await bot.generate_chart(pair)
        
        # Convert to base64 for API response
        base64_chart = base64.b64encode(chart_buf.getvalue()).decode("utf-8")
        
        return {
            "pair": pair,
            "image": f"data:image/png;base64,{base64_chart}",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error generating chart for {pair}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate chart: {str(e)}")

@app.get("/backtest")
async def run_backtest(
    pair: str,
    start_date: str = None,
    end_date: str = None,
    timeframe: str = None
):
    """Run a backtest for a specific pair and time period"""
    if pair not in bot.pairs:
        raise HTTPException(status_code=404, detail=f"Pair {pair} not found")
    
    try:
        # Parse dates
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00')) if start_date else None
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00')) if end_date else None
        
        # Fetch historical data
        data = await bot.fetch_historical_data(
            pair=pair,
            timeframe=timeframe or bot.timeframe,
            start_date=start,
            end_date=end
        )
        
        if data.empty:
            return {"message": "No data available for the specified period"}
        
        # Calculate indicators
        data = bot.calculate_indicators(data)
        
        # Run backtest
        results = []
        in_trade = False
        current_trade = None
        
        for i in range(1, len(data) - 1):
            # Check for entry if not in trade
            if not in_trade:
                # Create a slice for entry condition check
                test_data = data.iloc[:i+2]
                signal = bot.check_entry_conditions(test_data)
                
                if signal:
                    in_trade = True
                    current_trade = signal
            
            # Check for exit if in trade
            elif in_trade:
                # Create a slice for exit condition check
                current_slice = data.iloc[i:i+1]
                exit_result = bot.check_trade_exits(current_trade, current_slice)
                
                if exit_result:
                    results.append(exit_result)
                    in_trade = False
                    current_trade = None
        
        # Calculate statistics
        if results:
            total_trades = len(results)
            winning_trades = sum(1 for r in results if r.profit > 0)
            total_profit = sum(r.profit for r in results)
            win_rate = (winning_trades / total_trades) * 100 if total_trades else 0
            
            return {
                "pair": pair,
                "timeframe": timeframe or bot.timeframe,
                "start_date": start.isoformat() if start else data.index[0].isoformat(),
                "end_date": end.isoformat() if end else data.index[-1].isoformat(),
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": total_trades - winning_trades,
                "win_rate": f"{win_rate:.2f}%",
                "total_profit": f"${total_profit:.2f}",
                "results": [r.dict() for r in results]
            }
        else:
            return {
                "pair": pair,
                "timeframe": timeframe or bot.timeframe,
                "start_date": start.isoformat() if start else data.index[0].isoformat(),
                "end_date": end.isoformat() if end else data.index[-1].isoformat(),
                "message": "No trade signals generated during this period"
            }
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Error running backtest for {pair}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run backtest: {str(e)}")

@app.post("/settings")
async def update_settings(
    risk_amount: Optional[float] = None,
    min_risk_reward: Optional[float] = None,
    daily_loss_limit: Optional[float] = None,
    timeframe: Optional[str] = None,
    pairs: Optional[List[str]] = None
):
    """Update bot settings"""
    try:
        if risk_amount is not None:
            bot.risk_amount = risk_amount
            
        if min_risk_reward is not None:
            bot.min_risk_reward = min_risk_reward
            
        if daily_loss_limit is not None:
            bot.daily_loss_limit = daily_loss_limit
            
        if timeframe is not None:
            bot.timeframe = timeframe
            # We need to restart the scheduler with new timeframe
            bot.scheduler.shutdown()
            bot.scheduler = AsyncIOScheduler()
            asyncio.create_task(bot.start_scheduler())
            
        if pairs is not None:
            bot.pairs = pairs
            
        return {
            "message": "Settings updated successfully",
            "current_settings": {
                "risk_amount": bot.risk_amount,
                "min_risk_reward": bot.min_risk_reward,
                "daily_loss_limit": bot.daily_loss_limit,
                "timeframe": bot.timeframe,
                "pairs": bot.pairs
            }
        }
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("Starting Forex Trading Signal Bot")
    
    # Start the scheduler
    await bot.start_scheduler()
    
    # Send startup notification
    await bot.send_telegram_message(
        "🚀 *Forex Trading Signal Bot Started*\n"
        f"Monitoring {len(bot.pairs)} currency pairs on {bot.timeframe} timeframe"
    )
    
    # Perform initial scan
    asyncio.create_task(bot.run_scanner())

@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Forex Trading Signal Bot")
    
    # Shutdown the scheduler
    bot.scheduler.shutdown()
    
    # Send shutdown notification
    try:
        await bot.send_telegram_message("⚠️ *Forex Trading Signal Bot Stopped*")
    except:
        pass

# Run the application
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

