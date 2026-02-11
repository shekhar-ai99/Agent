#!/usr/bin/env python3
"""
TradingPlatform Backend API - Reference Implementation

This is a stub/reference implementation showing required endpoints
for the React UI. Adapt this to your actual backend.

Run: python fastapi_stub.py
Then: Access http://localhost:8000/docs for API docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
import random
import json
from pathlib import Path

app = FastAPI(title="TradingPlatform API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Models
# ============================================================================

class BacktestRequest(BaseModel):
    market: str
    exchange: str
    instrument: str
    timeframe: str
    capital: float
    risk_per_trade: float
    start_date: str
    end_date: str

class SimulationRequest(BaseModel):
    market: str
    exchange: str
    instrument: str
    timeframe: str
    capital: float
    risk_per_trade: float

class RunResponse(BaseModel):
    run_id: str
    message: str

class Trade(BaseModel):
    trade_id: int
    strategy: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    quantity: int
    pnl: float
    regime: str
    volatility: str
    session: str
    day: str

# ============================================================================
# Utility Functions
# ============================================================================

def generate_run_id():
    """Generate unique run ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{random.randint(1000, 9999)}"

def generate_equity_curve(capital: float, days: int = 30) -> list:
    """Generate sample equity curve"""
    curve = []
    current_equity = capital
    current_date = datetime.now() - timedelta(days=days)
    
    for _ in range(days):
        daily_change = random.uniform(-2000, 3000)
        current_equity = max(current_equity + daily_change, capital * 0.5)
        curve.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "equity": round(current_equity, 2)
        })
        current_date += timedelta(days=1)
    
    return curve

def generate_drawdown_curve(days: int = 30) -> list:
    """Generate sample drawdown curve"""
    curve = []
    current_date = datetime.now() - timedelta(days=days)
    current_dd = 0
    
    for _ in range(days):
        current_dd = max(current_dd - random.uniform(0, 2), -12)
        curve.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "drawdown": round(current_dd, 2)
        })
        current_date += timedelta(days=1)
    
    return curve

def generate_trades(capital: float, num_trades: int = 50) -> list:
    """Generate sample trades"""
    trades = []
    strategies = ["RSI_MeanReversion", "MA_Crossover", "BollingerBands", "SuperTrend"]
    regimes = ["TRENDING", "RANGING", "VOLATILE"]
    volatilities = ["LOW", "MEDIUM", "HIGH"]
    sessions = ["MORNING", "MIDDAY", "AFTERNOON"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    
    current_date = datetime.now() - timedelta(days=30)
    
    for i in range(num_trades):
        entry_date = current_date + timedelta(days=i % 25, hours=random.randint(9, 15))
        exit_date = entry_date + timedelta(hours=random.randint(1, 8))
        
        entry_price = random.uniform(22000, 24000)
        exit_price = entry_price * random.uniform(0.99, 1.01)
        qty = random.randint(10, 100)
        pnl = round((exit_price - entry_price) * qty, 2)
        
        trades.append({
            "trade_id": i + 1,
            "strategy": random.choice(strategies),
            "entry_time": entry_date.isoformat(),
            "entry_price": round(entry_price, 2),
            "exit_time": exit_date.isoformat(),
            "exit_price": round(exit_price, 2),
            "quantity": qty,
            "pnl": pnl,
            "regime": random.choice(regimes),
            "volatility": random.choice(volatilities),
            "session": random.choice(sessions),
            "day": random.choice(days)
        })
    
    return trades

def generate_results(config: dict) -> dict:
    """Generate complete backtest results"""
    capital = config.get("capital", 100000)
    trades = generate_trades(capital, num_trades=random.randint(40, 100))
    
    # Calculate metrics
    closed_trades = [t for t in trades if t.get("pnl")]
    wins = [t for t in closed_trades if t["pnl"] > 0]
    losses = [t for t in closed_trades if t["pnl"] < 0]
    
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = (len(wins) / len(closed_trades) * 100) if closed_trades else 0
    max_dd = -12.5
    sharpe = 1.8
    
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Generate curves
    days = (datetime.strptime(config.get("end_date", "2025-12-31"), "%Y-%m-%d") - 
            datetime.strptime(config.get("start_date", "2025-01-01"), "%Y-%m-%d")).days
    
    days = max(days, 1)
    
    equity_curve = generate_equity_curve(capital, min(days, 250))
    drawdown_curve = generate_drawdown_curve(min(days, 250))
    
    # Trades per day
    daily_counts = {}
    for trade in trades:
        day = datetime.fromisoformat(trade["entry_time"]).strftime("%Y-%m-%d")
        daily_counts[day] = daily_counts.get(day, 0) + 1
    
    trades_per_day = [
        {"day": day, "count": count}
        for day, count in sorted(daily_counts.items())
    ]
    
    return {
        "run_id": generate_run_id(),
        "status": "completed",
        "total_pnl": total_pnl,
        "net_pnl": total_pnl,
        "return_pct": round((total_pnl / capital) * 100, 2),
        "win_rate": round(win_rate, 2),
        "num_trades": len(closed_trades),
        "max_drawdown": max_dd,
        "sharpe_ratio": sharpe,
        "profit_factor": round(profit_factor, 2),
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
        "trades_per_day": trades_per_day,
        "pnl_trades": trades,
        "trades": trades
    }

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def read_root():
    """Health check"""
    return {"message": "TradingPlatform API is running"}

@app.get("/api/health")
def health():
    """Health check with details"""
    return {
        "status": "ok",
        "service": "TradingPlatform API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/run/backtest", response_model=RunResponse)
def run_backtest(request: BacktestRequest):
    """Start a backtest"""
    run_id = generate_run_id()
    
    print(f"Starting backtest: {run_id}")
    print(f"  Market: {request.market}")
    print(f"  Instrument: {request.instrument}")
    print(f"  Period: {request.start_date} to {request.end_date}")
    
    return RunResponse(
        run_id=run_id,
        message="Backtest started successfully"
    )

@app.post("/api/run/simulation", response_model=RunResponse)
def run_simulation(request: SimulationRequest):
    """Start a simulation"""
    run_id = generate_run_id()
    
    print(f"Starting simulation: {run_id}")
    print(f"  Market: {request.market}")
    print(f"  Instrument: {request.instrument}")
    
    return RunResponse(
        run_id=run_id,
        message="Simulation started successfully"
    )

@app.get("/api/results/{run_id}/status")
def get_run_status(run_id: str):
    """Check run status"""
    # Simulate status progression
    return {
        "run_id": run_id,
        "status": "completed",
        "progress": 100
    }

@app.get("/api/results/{run_id}")
def get_results(run_id: str):
    """Get backtest results"""
    try:
        # Generate sample results
        results = generate_results({
            "market": "india",
            "exchange": "nse",
            "instrument": "NIFTY",
            "timeframe": "5m",
            "capital": 100000,
            "start_date": "2025-01-01",
            "end_date": "2025-12-31"
        })
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{run_id}/trades")
def get_trades(run_id: str):
    """Get trade details"""
    try:
        trades = generate_trades(100000, num_trades=random.randint(40, 100))
        return {"trades": trades}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/strategy_ranking.json")
def get_strategy_rankings():
    """Get strategy rankings"""
    try:
        strategies = [
            {
                "rank": 1,
                "name": "RSI_MeanReversion",
                "market": "india",
                "win_rate": 58.5,
                "profit_factor": 1.8,
                "avg_pnl": 450.0,
                "total_pnl": 54000.0,
                "max_drawdown": -4.2,
                "trades_count": 120,
                "days": ["Monday", "Tuesday", "Wednesday"],
                "sessions": ["MORNING", "MIDDAY"],
                "regimes": ["TRENDING", "RANGING"],
                "volatilities": ["LOW", "MEDIUM"]
            },
            {
                "rank": 2,
                "name": "MA_Crossover",
                "market": "india",
                "win_rate": 52.3,
                "profit_factor": 1.5,
                "avg_pnl": 380.0,
                "total_pnl": 45600.0,
                "max_drawdown": -6.1,
                "trades_count": 120,
                "days": ["Monday", "Friday"],
                "sessions": ["MORNING"],
                "regimes": ["TRENDING"],
                "volatilities": ["LOW", "MEDIUM", "HIGH"]
            },
            {
                "rank": 3,
                "name": "BollingerBands",
                "market": "india",
                "win_rate": 48.9,
                "profit_factor": 1.2,
                "avg_pnl": 250.0,
                "total_pnl": 22500.0,
                "max_drawdown": -8.5,
                "trades_count": 90,
                "days": ["Tuesday", "Wednesday", "Thursday"],
                "sessions": ["AFTERNOON"],
                "regimes": ["RANGING"],
                "volatilities": ["MEDIUM", "HIGH"]
            }
        ]
        
        return {"strategies": strategies}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting TradingPlatform API...")
    print("📍 Available at: http://localhost:8000")
    print("📚 API Docs at: http://localhost:8000/docs")
    print("💬 ReDoc at: http://localhost:8000/redoc")
    uvicorn.run(app, host="0.0.0.0", port=8000)
