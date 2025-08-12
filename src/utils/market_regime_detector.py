import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
from dataclasses import dataclass, asdict
from enum import Enum
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from loguru import logger

class RegimeType(Enum):
    """Enum for market regime types"""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    RECOVERY = "recovery"
    DISTRIBUTION = "distribution"

@dataclass
class RegimeMetrics:
    """Data class for regime metrics"""
    regime: RegimeType
    trend_strength: float
    volatility: float
    momentum: float
    volume_profile: float
    confidence: float
    regime_duration: int
    support_resistance_strength: float
    mean_reversion_tendency: float
    regime_probability: Dict[str, float]
    timestamp: Optional[pd.Timestamp] = None

class TechnicalIndicators:
    """Technical analysis indicators for regime detection"""
    
    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_position(prices: np.ndarray, period: int = 20, std_dev: int = 2) -> float:
        """Calculate position within Bollinger Bands"""
        if len(prices) < period:
            return 0.5
        
        recent_prices = prices[-period:]
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        current_price = prices[-1]
        
        if upper_band == lower_band:
            return 0.5
        
        return (current_price - lower_band) / (upper_band - lower_band)
    
    @staticmethod
    def macd_signal(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """Calculate MACD and signal line"""
        if len(prices) < slow + signal:
            return 0.0, 0.0
        
        ema_fast = TechnicalIndicators._ema(prices, fast)
        ema_slow = TechnicalIndicators._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Simple moving average for signal line (simplified)
        if len(prices) >= signal:
            signal_line = np.mean([macd_line] * min(signal, len(prices)))
        else:
            signal_line = macd_line
        
        return macd_line, signal_line
    
    @staticmethod
    def _ema(prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema

class MarketRegimeDetector:
    """
    Market Regime Detector with multiple algorithms and features
    
    Features:
    - Multiple regime detection algorithms
    - Dynamic threshold adjustment
    - Confidence scoring with multiple factors
    - Regime persistence tracking
    - Volume analysis integration
    - Technical indicators integration
    - Machine learning clustering
    - Risk-adjusted metrics
    """
    
    def __init__(
        self,
        lookback_period: int = 60,
        short_period: int = 20,
        volatility_window: int = 30,
        min_regime_duration: int = 5,
        enable_ml_clustering: bool = True,
        enable_volume_analysis: bool = False,
        confidence_threshold: float = 0.6
    ):
        self.lookback_period = lookback_period
        self.short_period = short_period
        self.volatility_window = volatility_window
        self.min_regime_duration = min_regime_duration
        self.enable_ml_clustering = enable_ml_clustering
        self.enable_volume_analysis = enable_volume_analysis
        self.confidence_threshold = confidence_threshold
        
        # State tracking
        self.current_regime = RegimeType.SIDEWAYS
        self.regime_start_time = 0
        self.regime_history = deque(maxlen=100)
        self.price_history = deque(maxlen=max(lookback_period * 2, 200))
        self.volume_history = deque(maxlen=max(lookback_period * 2, 200)) if enable_volume_analysis else None
        
        # Dynamic thresholds
        self.adaptive_thresholds = {
            'trend_bull': 0.05,
            'trend_bear': -0.05,
            'volatility_low': 0.02,
            'volatility_high': 0.05
        }
        
        # ML components
        if enable_ml_clustering:
            self.scaler = StandardScaler()
            self.kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            self._ml_fitted = False
        
        self.indicators = TechnicalIndicators()
        
        logger.info(f"Advanced Market Regime Detector initialized with lookback={lookback_period}")
    
    def update_price(self, price: float, volume: Optional[float] = None, timestamp: Optional[pd.Timestamp] = None):
        """Update with new price data (for streaming/real-time use)"""
        self.price_history.append(price)
        
        if self.enable_volume_analysis and volume is not None:
            self.volume_history.append(volume)
        
        # Update regime if enough data
        if len(self.price_history) >= self.short_period:
            prices_array = np.array(list(self.price_history))
            volumes_array = np.array(list(self.volume_history)) if self.volume_history else None
            
            new_regime_metrics = self.detect_regime_comprehensive(prices_array, volumes_array)
            
            # Check for regime change
            if new_regime_metrics.regime != self.current_regime:
                if new_regime_metrics.confidence >= self.confidence_threshold:
                    self._regime_change(new_regime_metrics, timestamp)
            
            self.regime_history.append(new_regime_metrics)
    
    def _regime_change(self, new_metrics: RegimeMetrics, timestamp: Optional[pd.Timestamp]):
        """Handle regime change"""
        old_regime = self.current_regime
        self.current_regime = new_metrics.regime
        self.regime_start_time = len(self.price_history)
        
        logger.info(f"Regime change detected: {old_regime.value} -> {new_metrics.regime.value} "
                        f"(confidence: {new_metrics.confidence:.2f})")
    
    def detect_regime_comprehensive(
        self, 
        prices: np.ndarray, 
        volumes: Optional[np.ndarray] = None
    ) -> RegimeMetrics:
        """
        Comprehensive regime detection using multiple methods
        """
        if len(prices) < self.short_period:
            return self._default_regime_metrics()
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds(prices)
        
        # Calculate base metrics
        base_metrics = self._calculate_base_metrics(prices)
        
        # Calculate technical indicators
        technical_metrics = self._calculate_technical_metrics(prices)
        
        # Volume analysis
        volume_metrics = self._calculate_volume_metrics(volumes) if volumes is not None else {}
        
        # ML-based regime detection
        ml_regime_probs = self._ml_regime_detection(prices) if self.enable_ml_clustering else {}
        
        # Combine all methods for final regime determination
        final_regime, confidence = self._combine_regime_signals(
            base_metrics, technical_metrics, volume_metrics, ml_regime_probs
        )
        
        # Calculate additional metrics
        regime_duration = len(self.price_history) - self.regime_start_time if hasattr(self, 'regime_start_time') else 0
        support_resistance = self._calculate_support_resistance_strength(prices)
        mean_reversion = self._calculate_mean_reversion_tendency(prices)
        
        return RegimeMetrics(
            regime=final_regime,
            trend_strength=base_metrics['trend_strength'],
            volatility=base_metrics['volatility'],
            momentum=base_metrics['momentum'],
            volume_profile=volume_metrics.get('volume_trend', 0.0),
            confidence=confidence,
            regime_duration=regime_duration,
            support_resistance_strength=support_resistance,
            mean_reversion_tendency=mean_reversion,
            regime_probability=ml_regime_probs,
            timestamp=pd.Timestamp.now()
        )
    
    def _calculate_base_metrics(self, prices: np.ndarray) -> Dict:
        """Calculate basic trend and volatility metrics"""
        recent_prices = prices[-self.lookback_period:] if len(prices) >= self.lookback_period else prices
        
        # Returns calculation
        returns = np.diff(recent_prices) / recent_prices[:-1]
        
        # Trend strength (risk-adjusted)
        trend_strength = np.sum(returns)
        
        # Multiple volatility measures
        volatility = np.std(returns)
        volatility_garch = self._estimate_garch_volatility(returns)
        
        # Momentum indicators
        short_returns = returns[-self.short_period:] if len(returns) >= self.short_period else returns
        momentum = np.mean(short_returns) / (np.std(short_returns) + 1e-8)  # Sharpe-like ratio
        
        # Trend consistency
        positive_returns = np.sum(returns > 0)
        trend_consistency = positive_returns / len(returns) if len(returns) > 0 else 0.5
        
        return {
            'trend_strength': trend_strength,
            'volatility': volatility,
            'volatility_garch': volatility_garch,
            'momentum': momentum,
            'trend_consistency': trend_consistency,
            'returns': returns
        }
    
    def _calculate_technical_metrics(self, prices: np.ndarray) -> Dict:
        """Calculate technical indicator metrics"""
        try:
            rsi = self.indicators.rsi(prices)
            bb_position = self.indicators.bollinger_position(prices)
            macd, macd_signal = self.indicators.macd_signal(prices)
            
            # Moving average relationships
            sma_short = np.mean(prices[-self.short_period:]) if len(prices) >= self.short_period else prices[-1]
            sma_long = np.mean(prices[-self.lookback_period:]) if len(prices) >= self.lookback_period else np.mean(prices)
            ma_ratio = sma_short / sma_long - 1
            
            return {
                'rsi': rsi,
                'bb_position': bb_position,
                'macd': macd,
                'macd_signal': macd_signal,
                'ma_ratio': ma_ratio,
                'price_vs_sma': (prices[-1] / sma_long - 1) if sma_long != 0 else 0
            }
        except Exception as e:
            logger.warning(f"Error calculating technical metrics: {e}")
            return {
                'rsi': 50, 'bb_position': 0.5, 'macd': 0, 'macd_signal': 0,
                'ma_ratio': 0, 'price_vs_sma': 0
            }
    
    def _calculate_volume_metrics(self, volumes: np.ndarray) -> Dict:
        """Calculate volume-based metrics"""
        if volumes is None or len(volumes) < self.short_period:
            return {'volume_trend': 0.0, 'volume_volatility': 0.0}
        
        recent_volumes = volumes[-self.lookback_period:]
        volume_returns = np.diff(recent_volumes) / (recent_volumes[:-1] + 1e-8)
        
        return {
            'volume_trend': np.sum(volume_returns),
            'volume_volatility': np.std(volume_returns),
            'price_volume_correlation': np.corrcoef(
                np.diff(volumes[-len(recent_volumes):]), 
                np.diff(recent_volumes)
            )[0, 1] if len(recent_volumes) > 1 else 0
        }
    
    def _ml_regime_detection(self, prices: np.ndarray) -> Dict[str, float]:
        """Machine learning based regime detection using clustering"""
        try:
            if len(prices) < self.lookback_period:
                return {regime.value: 1/6 for regime in RegimeType}
            
            # Feature engineering
            features = self._extract_ml_features(prices)
            
            if not self._ml_fitted and len(self.price_history) >= self.lookback_period * 2:
                # Fit the model with historical data
                historical_features = []
                for i in range(self.lookback_period, len(self.price_history)):
                    hist_prices = np.array(list(self.price_history))[i-self.lookback_period:i]
                    historical_features.append(self._extract_ml_features(hist_prices))
                
                if len(historical_features) > 10:
                    X = np.array(historical_features)
                    X_scaled = self.scaler.fit_transform(X)
                    self.kmeans.fit(X_scaled)
                    self._ml_fitted = True
            
            if self._ml_fitted:
                features_scaled = self.scaler.transform([features])
                cluster = self.kmeans.predict(features_scaled)[0]
                
                # Map clusters to regimes (simplified mapping)
                regime_mapping = {
                    0: RegimeType.BULL,
                    1: RegimeType.BEAR,
                    2: RegimeType.SIDEWAYS,
                    3: RegimeType.VOLATILE,
                    4: RegimeType.RECOVERY,
                    5: RegimeType.DISTRIBUTION
                }
                
                probabilities = {regime.value: 0.1 for regime in RegimeType}
                probabilities[regime_mapping.get(cluster, RegimeType.SIDEWAYS).value] = 0.6
                
                return probabilities
            
        except Exception as e:
            logger.warning(f"ML regime detection failed: {e}")
        
        return {regime.value: 1/6 for regime in RegimeType}
    
    def _extract_ml_features(self, prices: np.ndarray) -> List[float]:
        """Extract features for ML model"""
        returns = np.diff(prices) / prices[:-1]
        
        features = [
            np.mean(returns),  # trend
            np.std(returns),   # volatility
            stats.skew(returns),  # skewness
            stats.kurtosis(returns),  # kurtosis
            self.indicators.rsi(prices) / 100,  # normalized RSI
            self.indicators.bollinger_position(prices),  # BB position
            np.sum(returns > 0) / len(returns),  # positive return ratio
            np.max(returns),   # max return
            np.min(returns),   # min return
            np.std(prices) / np.mean(prices)  # coefficient of variation
        ]
        
        return features
    
    def _combine_regime_signals(
        self, 
        base_metrics: Dict, 
        technical_metrics: Dict, 
        volume_metrics: Dict,
        ml_probs: Dict
    ) -> Tuple[RegimeType, float]:
        """Combine multiple signals to determine final regime"""
        
        # Base regime detection
        trend_strength = base_metrics['trend_strength']
        volatility = base_metrics['volatility']
        momentum = base_metrics['momentum']
        
        # Technical confirmation
        rsi = technical_metrics['rsi']
        bb_position = technical_metrics['bb_position']
        ma_ratio = technical_metrics['ma_ratio']
        
        # Regime scoring
        regime_scores = {regime: 0.0 for regime in RegimeType}
        
        # Trend-based scoring
        if trend_strength > self.adaptive_thresholds['trend_bull']:
            regime_scores[RegimeType.BULL] += 3
            if rsi > 50 and ma_ratio > 0:
                regime_scores[RegimeType.BULL] += 2
        elif trend_strength < self.adaptive_thresholds['trend_bear']:
            regime_scores[RegimeType.BEAR] += 3
            if rsi < 50 and ma_ratio < 0:
                regime_scores[RegimeType.BEAR] += 2
        
        # Volatility-based scoring
        if volatility > self.adaptive_thresholds['volatility_high']:
            regime_scores[RegimeType.VOLATILE] += 2
        elif volatility < self.adaptive_thresholds['volatility_low']:
            if abs(trend_strength) < 0.02:
                regime_scores[RegimeType.SIDEWAYS] += 3
        
        # Technical indicator scoring
        if 30 < rsi < 70 and 0.2 < bb_position < 0.8:
            regime_scores[RegimeType.SIDEWAYS] += 1
        
        if rsi < 30 and trend_strength < 0:
            regime_scores[RegimeType.RECOVERY] += 1
        
        if rsi > 70 and trend_strength > 0:
            regime_scores[RegimeType.DISTRIBUTION] += 1
        
        # ML probability integration
        if ml_probs:
            for regime_name, prob in ml_probs.items():
                try:
                    regime = RegimeType(regime_name)
                    regime_scores[regime] += prob * 2
                except ValueError:
                    pass
        
        # Find best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        max_score = regime_scores[best_regime]
        
        # Calculate confidence
        total_score = sum(regime_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0
        
        # Ensure minimum confidence for regime changes
        if confidence < self.confidence_threshold:
            confidence = confidence * 0.8  # Reduce confidence for uncertain cases
        
        return best_regime, min(confidence, 1.0)
    
    def _update_adaptive_thresholds(self, prices: np.ndarray):
        """Update thresholds based on market conditions"""
        if len(prices) < self.lookback_period:
            return
        
        returns = np.diff(prices) / prices[:-1]
        recent_vol = np.std(returns[-self.volatility_window:])
        
        # Adjust thresholds based on current volatility regime
        vol_multiplier = max(0.5, min(2.0, recent_vol / 0.02))
        
        self.adaptive_thresholds.update({
            'trend_bull': 0.05 * vol_multiplier,
            'trend_bear': -0.05 * vol_multiplier,
            'volatility_low': 0.02 * vol_multiplier,
            'volatility_high': 0.05 * vol_multiplier
        })
    
    def _estimate_garch_volatility(self, returns: np.ndarray) -> float:
        """Simplified GARCH-like volatility estimation"""
        if len(returns) < 5:
            return np.std(returns)
        
        # Simple EWMA volatility
        weights = np.exp(-0.1 * np.arange(len(returns))[::-1])
        weighted_returns = returns * weights
        return np.sqrt(np.sum(weighted_returns ** 2) / np.sum(weights))
    
    def _calculate_support_resistance_strength(self, prices: np.ndarray) -> float:
        """Calculate support/resistance level strength"""
        if len(prices) < 20:
            return 0.0
        
        # Find local maxima and minima
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append(prices[i])
        
        # Calculate clustering of support/resistance levels
        if len(highs) < 2 or len(lows) < 2:
            return 0.0
        
        # Simplified strength calculation
        current_price = prices[-1]
        resistance_distance = min([abs(h - current_price) for h in highs]) if highs else float('inf')
        support_distance = min([abs(l - current_price) for l in lows]) if lows else float('inf')
        
        # Normalize to 0-1 scale
        min_distance = min(resistance_distance, support_distance)
        price_range = np.max(prices) - np.min(prices)
        
        if price_range == 0:
            return 0.0
        
        return max(0, 1 - (min_distance / price_range))
    
    def _calculate_mean_reversion_tendency(self, prices: np.ndarray) -> float:
        """Calculate mean reversion tendency"""
        if len(prices) < self.short_period:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate autocorrelation of returns (lag-1)
        if len(returns) < 2:
            return 0.0
        
        return -np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0.0
    
    def _default_regime_metrics(self) -> RegimeMetrics:
        """Return default regime metrics when insufficient data"""
        return RegimeMetrics(
            regime=RegimeType.SIDEWAYS,
            trend_strength=0.0,
            volatility=0.0,
            momentum=0.0,
            volume_profile=0.0,
            confidence=0.0,
            regime_duration=0,
            support_resistance_strength=0.0,
            mean_reversion_tendency=0.0,
            regime_probability={regime.value: 1/6 for regime in RegimeType}
        )
    
    def get_regime_summary(self, prices: np.ndarray, volumes: Optional[np.ndarray] = None) -> Dict:
        """Get comprehensive regime summary"""
        metrics = self.detect_regime_comprehensive(prices, volumes)
        
        return {
            'current_regime': metrics.regime.value,
            'confidence': f"{metrics.confidence:.2%}",
            'trend_strength': f"{metrics.trend_strength:.4f}",
            'volatility': f"{metrics.volatility:.4f}",
            'momentum': f"{metrics.momentum:.4f}",
            'regime_duration': metrics.regime_duration,
            'support_resistance': f"{metrics.support_resistance_strength:.2f}",
            'mean_reversion': f"{metrics.mean_reversion_tendency:.2f}",
            'regime_probabilities': {k: f"{v:.1%}" for k, v in metrics.regime_probability.items()},
            'adaptive_thresholds': self.adaptive_thresholds,
            'recommendation': self._get_trading_recommendation(metrics)
        }
    
    def _get_trading_recommendation(self, metrics: RegimeMetrics) -> str:
        """Generate trading recommendation based on regime"""
        regime = metrics.regime
        confidence = metrics.confidence
        
        if confidence < 0.6:
            return "LOW CONFIDENCE - Wait for clearer signals"
        
        recommendations = {
            RegimeType.BULL: "BULLISH - Consider long positions, trend-following strategies",
            RegimeType.BEAR: "BEARISH - Consider short positions, defensive strategies", 
            RegimeType.SIDEWAYS: "NEUTRAL - Range-bound strategies, mean reversion",
            RegimeType.VOLATILE: "HIGH VOLATILITY - Risk management priority, smaller positions",
            RegimeType.RECOVERY: "RECOVERY - Potential buying opportunity, gradual accumulation",
            RegimeType.DISTRIBUTION: "DISTRIBUTION - Potential selling opportunity, profit-taking"
        }
        
        return recommendations.get(regime, "UNCERTAIN - Monitor closely")