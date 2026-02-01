"""Simulated broker for paper trading."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger("algotrade.execution")


@dataclass
class Position:
    """A position in a security."""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float = 0.0
    
    @property
    def value(self) -> float:
        return abs(self.quantity) * self.current_price
    
    @property
    def pnl(self) -> float:
        if self.quantity > 0:
            return self.quantity * (self.current_price - self.avg_price)
        return abs(self.quantity) * (self.avg_price - self.current_price)


@dataclass
class Trade:
    """A completed trade."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    commission: float = 0.0


class SimulatedBroker:
    """Paper trading broker with simulated execution."""
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_share: float = 0.01,
        slippage_pct: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct
        
        self.positions: Dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.prices: Dict[str, float] = {}
    
    @property
    def equity(self) -> float:
        pos_value = sum(p.value for p in self.positions.values())
        return self.cash + pos_value
    
    @property
    def unrealized_pnl(self) -> float:
        return sum(p.pnl for p in self.positions.values())
    
    def update_price(self, symbol: str, price: float):
        """Update current price for a symbol."""
        self.prices[symbol] = price
        if symbol in self.positions:
            self.positions[symbol].current_price = price
    
    def buy(self, symbol: str, quantity: float) -> Optional[Trade]:
        """Execute a buy order."""
        price = self.prices.get(symbol, 0)
        if price <= 0:
            logger.warning(f"No price for {symbol}")
            return None
        
        # Apply slippage (buy at higher price)
        exec_price = price * (1 + self.slippage_pct)
        commission = quantity * self.commission_per_share
        total_cost = quantity * exec_price + commission
        
        if total_cost > self.cash:
            logger.warning(f"Insufficient funds: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return None
        
        self.cash -= total_cost
        self._update_position(symbol, quantity, exec_price)
        
        trade = Trade(symbol, "buy", quantity, exec_price, commission=commission)
        self.trades.append(trade)
        logger.info(f"BUY {quantity:.2f} {symbol} @ ${exec_price:.2f}")
        return trade
    
    def sell(self, symbol: str, quantity: float) -> Optional[Trade]:
        """Execute a sell order."""
        price = self.prices.get(symbol, 0)
        if price <= 0:
            logger.warning(f"No price for {symbol}")
            return None
        
        # Apply slippage (sell at lower price)
        exec_price = price * (1 - self.slippage_pct)
        commission = quantity * self.commission_per_share
        proceeds = quantity * exec_price - commission
        
        self.cash += proceeds
        self._update_position(symbol, -quantity, exec_price)
        
        trade = Trade(symbol, "sell", quantity, exec_price, commission=commission)
        self.trades.append(trade)
        logger.info(f"SELL {quantity:.2f} {symbol} @ ${exec_price:.2f}")
        return trade
    
    def close_position(self, symbol: str) -> Optional[Trade]:
        """Close entire position in a symbol."""
        pos = self.positions.get(symbol)
        if not pos or pos.quantity == 0:
            return None
        
        if pos.quantity > 0:
            return self.sell(symbol, pos.quantity)
        return self.buy(symbol, abs(pos.quantity))
    
    def _update_position(self, symbol: str, qty_change: float, price: float):
        """Update or create position after a trade."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol, 0, 0, price)
        
        pos = self.positions[symbol]
        old_qty = pos.quantity
        new_qty = old_qty + qty_change
        
        if new_qty == 0:
            del self.positions[symbol]
        elif (old_qty >= 0 and new_qty > 0) or (old_qty <= 0 and new_qty < 0):
            # Adding to position - update avg price
            old_value = abs(old_qty) * pos.avg_price
            new_value = abs(qty_change) * price
            pos.avg_price = (old_value + new_value) / abs(new_qty)
            pos.quantity = new_qty
            pos.current_price = price
        else:
            # Position flipped
            pos.quantity = new_qty
            pos.avg_price = price
            pos.current_price = price
    
    def reset(self):
        """Reset to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.prices.clear()


class RiskManager:
    """Simple risk management with position limits and drawdown checks."""
    
    def __init__(
        self,
        max_position_pct: float = 0.25,
        max_drawdown: float = 0.2,
        max_daily_loss: float = 0.05,
    ):
        self.max_position_pct = max_position_pct
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        
        self.high_water_mark = 0.0
        self.day_start_equity = 0.0
        self.kill_switch = False
    
    def check_order(self, broker: SimulatedBroker, symbol: str, quantity: float, is_buy: bool) -> tuple[bool, str]:
        """Check if an order is allowed. Returns (allowed, reason)."""
        if self.kill_switch:
            return False, "Kill switch triggered"
        
        equity = broker.equity
        self.high_water_mark = max(self.high_water_mark, equity)
        
        # Check drawdown
        drawdown = (self.high_water_mark - equity) / self.high_water_mark if self.high_water_mark > 0 else 0
        if drawdown >= self.max_drawdown:
            self.kill_switch = True
            return False, f"Max drawdown exceeded: {drawdown:.1%}"
        
        # Check position size
        if is_buy:
            price = broker.prices.get(symbol, 0)
            order_value = quantity * price
            pos_value = broker.positions.get(symbol, Position(symbol, 0, 0)).value
            total_pos = order_value + pos_value
            if total_pos / equity > self.max_position_pct:
                return False, f"Position too large: {total_pos/equity:.1%} > {self.max_position_pct:.1%}"
        
        return True, ""
    
    def reset(self):
        self.high_water_mark = 0.0
        self.day_start_equity = 0.0
        self.kill_switch = False
