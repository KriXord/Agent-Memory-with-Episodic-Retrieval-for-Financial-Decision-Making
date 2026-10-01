import matplotlib
matplotlib.use('Agg')
import talib
import pandas as pd
import matplotlib.pyplot as plt
import talib
import numpy as np
from langchain_core.tools import tool
from typing import Annotated, Optional
import mplfinance as mpf
import base64
import io
import mplfinance as mpf 
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity


# helper function for trending graph
def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y.iloc[pivot]

    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y.iloc[pivot]
)


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 



def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)

def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
    # compute sum of differences between line and prices, 
    # return negative val if invalid 
    
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y.iloc[pivot]

    line_vals = slope * np.arange(len(y)) + intercept
     
    diffs = line_vals - y
    
    # Check to see if the line is valid, return -1 if it is not valid.
    if support and diffs.max() > 1e-5:
        return -1.0
    elif not support and diffs.min() < -1e-5:
        return -1.0

    # Squared sum of diffs between data and line 
    err = (diffs ** 2.0).sum()
    return err

def optimize_slope(support: bool, pivot:int , init_slope: float, y: np.array):
    
    # Amount to change slope by. Multiplyed by opt_step
    slope_unit = (y.max() - y.min()) / len(y) 
    
    # Optmization variables
    opt_step = 1.0
    min_step = 0.0001
    curr_step = opt_step # current step
    
    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(support, pivot, init_slope, y)
    assert(best_err >= 0.0) # Shouldn't ever fail with initial slope

    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount
            # to see if error increases/decreases. 
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(support, pivot, slope_change, y)
            derivative = test_err - best_err;
            
            # If increasing by a small amount fails, 
            # try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(support, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0: # Derivative failed, give up
                raise Exception("Derivative failed. Check your data. ")

            get_derivative = False

        if derivative > 0.0: # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else: # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step
        

        test_err = check_trend_line(support, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err: 
            # slope failed/didn't reduce error
            curr_step *= 0.5 # Reduce step size
        else: # test slope reduced error
            best_err = test_err 
            best_slope = test_slope
            get_derivative = True # Recompute derivative
    
    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y.iloc[pivot]
)


def fit_trendlines_single(data: np.array):
    # find line of best fit (least squared) 
    # coefs[0] = slope,  coefs[1] = intercept 
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)

    # Get points of line.
    line_points = coefs[0] * x + coefs[1]

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax() 
    lower_pivot = (data - line_points).argmin() 
   
    # Optimize the slope for both trend lines
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], data)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], data)

    return (support_coefs, resist_coefs) 



def fit_trendlines_high_low(high: np.array, low: np.array, close: np.array):
    x = np.arange(len(close))
    coefs = np.polyfit(x, close, 1)
    # coefs[0] = slope,  coefs[1] = intercept
    line_points = coefs[0] * x + coefs[1]
    upper_pivot = (high - line_points).argmax() 
    lower_pivot = (low - line_points).argmin() 
    
    support_coefs = optimize_slope(True, lower_pivot, coefs[0], low)
    resist_coefs = optimize_slope(False, upper_pivot, coefs[0], high)

    return (support_coefs, resist_coefs)

def get_line_points(candles, line_points):
    # Place line points in tuples for matplotlib finance
    # https://github.com/matplotlib/mplfinance/blob/master/examples/using_lines.ipynb
    idx = candles.index
    line_i = len(candles) - len(line_points)
    assert(line_i >= 0)
    points = []
    for i in range(line_i, len(candles)):
        points.append((idx[i], line_points[i - line_i]))
    return points


def split_line_into_segments(line_points):
    return [[line_points[i], line_points[i+1]] for i in range(len(line_points) - 1)]

class TechnicalTools:

    @staticmethod
    @tool
    def generate_anchored_vwap_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'."],
    anchor_date: Annotated[Optional[str], "Start date (YYYY-MM-DD or full datetime) to anchor VWAP from"] = None
    ) -> dict:
        """
        Generate a candlestick chart with Anchored VWAP indicator,
        save it locally as 'anchored_vwap_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.
            anchor_date (str): Date or datetime string to anchor VWAP from.
                            If None, will anchor from the first available candle.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Define anchor index
        if anchor_date is not None:
            anchor_date = pd.to_datetime(anchor_date)
            if anchor_date not in candles.index:
                # Use nearest available date
                anchor_date = candles.index[candles.index.get_indexer([anchor_date], method="nearest")[0]]
        else:
            anchor_date = candles.index[0]

        # Compute typical price
        typical_price = (candles["High"] + candles["Low"] + candles["Close"]) / 3

        # Compute cumulative values starting from anchor_date
        mask = candles.index >= anchor_date
        cum_vol = candles.loc[mask, "Volume"].cumsum()
        cum_tp_vol = (typical_price.loc[mask] * candles.loc[mask, "Volume"]).cumsum()

        vwap = cum_tp_vol / cum_vol

        # Place VWAP back into full index (NaN before anchor)
        vwap_full = pd.Series(index=candles.index, dtype=float)
        vwap_full.loc[mask] = vwap

        # Create addplot for VWAP
        apds = [
            mpf.make_addplot(vwap_full, color='purple', width=2, label=f'Anchored VWAP ({anchor_date.date()})')
        ]

        # Generate figure with Anchored VWAP
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            returnfig=True,
            figsize=(12, 8),
            block=False,
        )

        # Add titles and legend
        fig.suptitle(f'Candlestick Chart with Anchored VWAP (Anchor: {anchor_date.date()})', fontsize=14, y=0.95)
        axlist[0].set_title('Anchored VWAP Indicator', fontsize=12, pad=10)
        axlist[0].legend(loc='upper left')

        # Save fig locally
        fig.savefig(
            "anchored_vwap_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "anchored_vwap_image": img_b64,
            "anchored_vwap_image_description": f"Candlestick chart with Anchored VWAP line (purple) starting from {anchor_date.date()}. VWAP reflects the average traded price weighted by volume since the anchor point, helping identify fair value zones."
        }

    @staticmethod
    @tool
    def generate_vwap_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'."]
    ) -> dict:
        """
        Generate a candlestick chart with VWAP line from OHLCV data,
        save it locally as 'vwap_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Calculate VWAP using cumulative method
        # Calculate typical price for each period
        typical_price = (candles["High"] + candles["Low"] + candles["Close"]) / 3
        
        # Calculate price * volume
        price_volume = typical_price * candles["Volume"]
        
        # Calculate cumulative VWAP
        cumulative_pv = price_volume.cumsum()
        cumulative_volume = candles["Volume"].cumsum()
        
        # Calculate VWAP (avoid division by zero)
        vwap_series = cumulative_pv / cumulative_volume.replace(0, np.nan)
        vwap_series = vwap_series.ffill()  # Forward fill any NaN values

        # Create addplot for VWAP line
        apds = [
            mpf.make_addplot(vwap_series, color='purple', width=2, label="VWAP")
        ]

        # Generate figure with VWAP
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            returnfig=True,
            figsize=(12, 6),
            block=False,
        )

        # Add legend
        axlist[0].legend(loc='upper left')

        # Add title
        fig.suptitle('Candlestick Chart with VWAP', fontsize=14, y=0.95)

        # Save fig locally
        fig.savefig(
            "vwap_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "vwap_image": img_b64,
            "vwap_image_description": "Candlestick chart with VWAP (Volume Weighted Average Price) line. Purple line shows the cumulative volume-weighted average price from the beginning of the trading period."
        }

    @staticmethod
    @tool
    def generate_heiken_ashi_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."]
    ) -> dict:
        """
        Generate a Heiken-Ashi candlestick chart from traditional OHLCV data,
        save it locally as 'heiken_ashi_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Calculate Heiken-Ashi values
        ha_open = np.zeros(len(candles))
        ha_high = np.zeros(len(candles))
        ha_low = np.zeros(len(candles))
        ha_close = np.zeros(len(candles))
        
        for i in range(len(candles)):
            # Calculate Heiken-Ashi Close: (O + H + L + C) / 4
            ha_close[i] = (candles["Open"].iloc[i] + candles["High"].iloc[i] + 
                        candles["Low"].iloc[i] + candles["Close"].iloc[i]) / 4
            
            if i == 0:
                # First candle initialization
                ha_open[i] = (candles["Open"].iloc[i] + candles["Close"].iloc[i]) / 2
                ha_high[i] = candles["High"].iloc[i]
                ha_low[i] = candles["Low"].iloc[i]
            else:
                # Subsequent candles
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
                ha_high[i] = max(candles["High"].iloc[i], ha_open[i], ha_close[i])
                ha_low[i] = min(candles["Low"].iloc[i], ha_open[i], ha_close[i])

        # Create Heiken-Ashi DataFrame for plotting
        ha_data = pd.DataFrame({
            'Open': ha_open,
            'High': ha_high,
            'Low': ha_low,
            'Close': ha_close
        }, index=candles.index)

        # Create addplot lines to show original close price for comparison
        apds = [
            mpf.make_addplot(candles['Close'], color='blue', width=1, alpha=0.7, label="Original Close")
        ]

        # Generate Heiken-Ashi candlestick chart
        fig, axlist = mpf.plot(
            ha_data,  # Use Heiken-Ashi data instead of original candles
            type='candle',
            style='charles',
            addplot=apds,  # Overlay original close price
            returnfig=True,
            figsize=(12, 6),
            block=False,
        )

        # Add legend
        axlist[0].legend(loc='upper left')

        # Add title
        fig.suptitle('Heiken-Ashi Candlestick Chart with Original Close Price', fontsize=14, y=0.95)

        # Save fig locally
        fig.savefig(
            "heiken_ashi_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "heiken_ashi_image": img_b64,
            "heiken_ashi_image_description": "Heiken-Ashi candlestick chart showing smoothed price action with reduced noise. Blue line shows original close prices for comparison. Heiken-Ashi candles help identify trend direction more clearly."
        }
    
    @staticmethod
    @tool
    def generate_macd_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."],
    fastperiod: Annotated[int, "Fast EMA period"] = 12,
    slowperiod: Annotated[int, "Slow EMA period"] = 26,
    signalperiod: Annotated[int, "Signal line EMA period"] = 9
    ) -> dict:
        """
        Generate a candlestick chart with MACD indicator subplot from OHLCV data,
        save it locally as 'macd_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.
            fastperiod (int): Fast EMA period for MACD calculation.
            slowperiod (int): Slow EMA period for MACD calculation.
            signalperiod (int): Signal line EMA period for MACD calculation.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Calculate MACD using TA-Lib
        df = pd.DataFrame({'Close': candles['Close']})
        macd, macd_signal, macd_hist = talib.MACD(
            df["Close"], 
            fastperiod=fastperiod, 
            slowperiod=slowperiod, 
            signalperiod=signalperiod
        )

        # Create MACD DataFrame with proper index
        macd_data = pd.DataFrame({
            'MACD': macd,
            'Signal': macd_signal,
            'Histogram': macd_hist
        }, index=candles.index)

        # Create addplot for MACD components
        apds = [
            # MACD line (blue)
            mpf.make_addplot(macd_data['MACD'], color='blue', width=2, panel=1, ylabel='MACD', label='MACD'),
            # Signal line (red)
            mpf.make_addplot(macd_data['Signal'], color='red', width=2, panel=1, label='Signal'),
            # Histogram (green/red bars)
            mpf.make_addplot(macd_data['Histogram'], type='bar', color='gray', alpha=0.6, panel=1, label='Histogram')
        ]

        # Generate figure with MACD subplot
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            returnfig=True,
            figsize=(12, 8),  # Taller figure to accommodate MACD subplot
            panel_ratios=(3, 1),  # 3:1 ratio between main chart and MACD panel
            block=False,
        )

        # Add legends for both panels
        # MACD line labels are shown on their own panel.
        if axlist[2].get_legend_handles_labels()[0]:
            axlist[2].legend(loc='upper left')

        # Add horizontal line at zero for MACD panel
        axlist[1].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        # Add titles
        fig.suptitle(f'Candlestick Chart with MACD ({fastperiod},{slowperiod},{signalperiod})', fontsize=14, y=0.95)
        axlist[1].set_title('MACD Indicator', fontsize=12, pad=10)

        # Improve MACD panel formatting
        axlist[1].grid(True, alpha=0.3)
        
        # Color the histogram bars based on positive/negative values
        hist_colors = ['green' if x >= 0 else 'red' for x in macd_hist.dropna()]
        
        # Save fig locally
        fig.savefig(
            "macd_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "macd_image": img_b64,
            "macd_image_description": f"Candlestick chart with MACD indicator subplot using periods ({fastperiod},{slowperiod},{signalperiod}). Blue line shows MACD line, red line shows signal line, and gray bars show the MACD histogram. Zero line helps identify bullish/bearish crossovers."
        }
    
    @staticmethod
    @tool
    def generate_sma_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."],
    short_period: Annotated[int, "Short-term SMA period"] = 20,
    long_period: Annotated[int, "Long-term SMA period"] = 40
    ) -> dict:
        """
        Generate a candlestick chart with SMA indicators (short and long-term),
        save it locally as 'sma_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.
            short_period (int): Period for short-term SMA.
            long_period (int): Period for long-term SMA.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Calculate SMAs using TA-Lib
        df = pd.DataFrame({'Close': candles['Close']})
        sma_short = talib.SMA(df["Close"], timeperiod=short_period)
        sma_long = talib.SMA(df["Close"], timeperiod=long_period)

        # Create SMA DataFrame with proper index
        sma_data = pd.DataFrame({
            f'SMA_{short_period}': sma_short,
            f'SMA_{long_period}': sma_long
        }, index=candles.index)

        # Create addplots for SMA
        apds = [
            mpf.make_addplot(sma_data[f'SMA_{short_period}'], color='blue', width=2, label=f'SMA {short_period}'),
            mpf.make_addplot(sma_data[f'SMA_{long_period}'], color='red', width=2, label=f'SMA {long_period}')
        ]

        # Generate figure with SMAs on main chart
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            returnfig=True,
            figsize=(12, 8),
            block=False,
        )

        # Add titles and legend
        fig.suptitle(f'Candlestick Chart with SMA ({short_period}, {long_period})', fontsize=14, y=0.95)
        axlist[0].set_title('SMA Indicator', fontsize=12, pad=10)
        axlist[0].legend(loc='upper left')

        # Save fig locally
        fig.savefig(
            "sma_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "sma_image": img_b64,
            "sma_image_description": f"Candlestick chart with SMA indicators. Blue line shows the {short_period}-period SMA (short-term trend), and red line shows the {long_period}-period SMA (long-term trend). Crossovers between them help identify bullish and bearish signals."
        }

    @staticmethod
    @tool
    def generate_stochastic_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."],
    fastk_period: Annotated[int, "Number of periods for %K calculation"] = 14,
    slowk_period: Annotated[int, "Smoothing for %K"] = 3,
    slowd_period: Annotated[int, "Smoothing for %D"] = 3
    ) -> dict:
        """
        Generate a candlestick chart with Stochastic Oscillator subplot from OHLCV data,
        save it locally as 'stochastic_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.
            fastk_period (int): Period for %K calculation.
            slowk_period (int): Smoothing period for %K.
            slowd_period (int): Smoothing period for %D.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Calculate Stochastic Oscillator using TA-Lib
        slowk, slowd = talib.STOCH(
            candles["High"],
            candles["Low"],
            candles["Close"],
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowk_matype=0,
            slowd_period=slowd_period,
            slowd_matype=0
        )

        # Create stochastic DataFrame with proper index
        stoch_data = pd.DataFrame({
            '%K': slowk,
            '%D': slowd
        }, index=candles.index)

        # Create addplots for %K and %D
        apds = [
            mpf.make_addplot(stoch_data['%K'], color='blue', width=2, panel=1, ylabel='Stochastic', label='%K'),
            mpf.make_addplot(stoch_data['%D'], color='red', width=2, panel=1, label='%D'),
        ]

        # Generate figure with Stochastic Oscillator subplot
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            returnfig=True,
            figsize=(12, 8),
            panel_ratios=(3, 1),
            block=False,
        )

        # Add horizontal lines at overbought/oversold
        axlist[1].axhline(y=80, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (80)')
        axlist[1].axhline(y=20, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (20)')
        axlist[1].axhline(y=50, color='gray', linestyle='-', linewidth=1, alpha=0.5)

        # Add titles
        fig.suptitle(
            f'Candlestick Chart with Stochastic Oscillator (%K={fastk_period}, %D={slowd_period})',
            fontsize=14,
            y=0.95
        )
        axlist[1].set_title('Stochastic Oscillator', fontsize=12, pad=10)

        # Improve formatting
        axlist[1].set_ylim(0, 100)
        axlist[1].grid(True, alpha=0.3)
        axlist[1].legend(loc='upper left')

        # Save fig locally
        fig.savefig(
            "stochastic_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "stochastic_image": img_b64,
            "stochastic_image_description": f"Candlestick chart with Stochastic Oscillator subplot using %K={fastk_period}, %D={slowd_period}. Blue line shows %K, red line shows %D. Horizontal lines at 80 and 20 indicate overbought and oversold zones."
        }

    @staticmethod
    @tool
    def generate_rsi_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."],
    timeperiod: Annotated[int, "RSI period length"] = 14
    ) -> dict:
        """
        Generate a candlestick chart with RSI indicator subplot from OHLCV data,
        save it locally as 'rsi_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.
            timeperiod (int): Period length for RSI calculation.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Calculate RSI using TA-Lib
        df = pd.DataFrame({'Close': candles['Close']})
        rsi = talib.RSI(df["Close"], timeperiod=timeperiod)

        # Create RSI DataFrame with proper index
        rsi_data = pd.DataFrame({'RSI': rsi}, index=candles.index)

        # Create addplot for RSI
        apds = [
            mpf.make_addplot(rsi_data['RSI'], color='blue', width=2, panel=1, ylabel='RSI', label='RSI')
        ]

        # Generate figure with RSI subplot
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            returnfig=True,
            figsize=(12, 8),
            panel_ratios=(3, 1),
            block=False,
        )

        # Add horizontal lines for RSI thresholds
        axlist[1].axhline(y=70, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Overbought (70)')
        axlist[1].axhline(y=30, color='green', linestyle='--', linewidth=1, alpha=0.7, label='Oversold (30)')
        axlist[1].axhline(y=50, color='gray', linestyle='-', linewidth=1, alpha=0.5)

        # Add titles
        fig.suptitle(f'Candlestick Chart with RSI ({timeperiod})', fontsize=14, y=0.95)
        axlist[1].set_title('RSI Indicator', fontsize=12, pad=10)

        # Improve RSI panel formatting
        axlist[1].set_ylim(0, 100)
        axlist[1].grid(True, alpha=0.3)
        axlist[1].legend(loc='upper left')

        # Save fig locally
        fig.savefig(
            "rsi_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "rsi_image": img_b64,
            "rsi_image_description": f"Candlestick chart with RSI indicator subplot using period {timeperiod}. Blue line shows RSI values, with horizontal lines at 70 (overbought), 30 (oversold), and 50 (neutral midpoint)."
        }

    @staticmethod
    @tool
    def generate_bollinger_bands_image(
    kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."],
    timeperiod: Annotated[int, "Period for SMA in Bollinger Bands"] = 20,
    nbdevup: Annotated[int, "Number of standard deviations above"] = 2,
    nbdevdn: Annotated[int, "Number of standard deviations below"] = 2
    ) -> dict:
        """
        Generate a candlestick chart with Bollinger Bands overlay from OHLCV data,
        save it locally as 'bollinger_bands_graph.png', and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary containing OHLCV data.
            timeperiod (int): SMA period for Bollinger Bands.
            nbdevup (int): Standard deviations above SMA for upper band.
            nbdevdn (int): Standard deviations below SMA for lower band.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Calculate Bollinger Bands using TA-Lib
        upperband, middleband, lowerband = talib.BBANDS(
            candles["Close"],
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn,
            matype=0
        )

        # Create Bollinger Bands DataFrame
        bb_data = pd.DataFrame({
            'Upper Band': upperband,
            'Middle Band': middleband,
            'Lower Band': lowerband
        }, index=candles.index)

        # Create addplots for bands
        apds = [
            mpf.make_addplot(bb_data['Upper Band'], color='red', width=1.5, label='Upper Band'),
            mpf.make_addplot(bb_data['Middle Band'], color='blue', width=1.5, label='Middle SMA'),
            mpf.make_addplot(bb_data['Lower Band'], color='green', width=1.5, label='Lower Band'),
        ]

        # Generate figure with Bollinger Bands overlay
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            returnfig=True,
            figsize=(12, 8),
            block=False,
        )

        # Add titles and legend
        fig.suptitle(
            f'Candlestick Chart with Bollinger Bands (Period={timeperiod}, ±{nbdevup}σ)',
            fontsize=14,
            y=0.95
        )
        axlist[0].set_title('Bollinger Bands Indicator', fontsize=12, pad=10)
        axlist[0].legend(loc='upper left')

        # Save fig locally
        fig.savefig(
            "bollinger_bands_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "bollinger_bands_image": img_b64,
            "bollinger_bands_image_description": f"Candlestick chart with Bollinger Bands overlay. Red = Upper Band (+{nbdevup}σ), Blue = Middle SMA ({timeperiod}-period), Green = Lower Band (-{nbdevdn}σ). Bands expand/contract with volatility, helping identify breakouts and reversals."
        }

    @staticmethod
    @tool
    def generate_fibonacci_image(
    kline_data: dict,
    auto_trend: bool = True,
    custom_high: Optional[float] = None,
    custom_low: Optional[float] = None
    ) -> dict:
        """
        Generate a candlestick chart with Fibonacci retracement levels,
        save it locally as 'fibonacci_graph.png', and return a base64-encoded image.
        """
        data = pd.DataFrame(kline_data)
        candles = data.copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Downsample data if it's too large to prevent image size issues
        max_points = 2000  # Reasonable limit for visualization
        if len(candles) > max_points:
            print(f"Dataset has {len(candles)} points, downsampling to {max_points} for visualization")
            step = max(1, len(candles) // max_points)
            candles = candles.iloc[::step].copy()

        # Determine swing high and low
        if custom_high is not None and custom_low is not None:
            swing_high, swing_low = custom_high, custom_low
        else:
            swing_high = float(candles["High"].max())
            swing_low = float(candles["Low"].min())

        # Safety: avoid zero division
        diff = swing_high - swing_low
        if diff == 0:
            raise ValueError("Swing high and low are equal; cannot compute Fibonacci levels.")

        # Detect trend if auto
        if auto_trend:
            if candles["Close"].iloc[-1] > candles["Close"].iloc[0]:
                trend = "up"
            else:
                trend = "down"
        else:
            trend = "up" if swing_low < swing_high else "down"

        # Compute Fibonacci retracement levels in the correct order (0 -> 100)
        ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        labels = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]

        if trend == "up":
            prices = [swing_low + r * diff for r in ratios]
        else:  # downtrend: measure from the HIGH downward
            prices = [swing_high - r * diff for r in ratios]

        # Preserve order as list of tuples so drawing/labels follow 0%->100%
        fib_levels = list(zip(labels, prices))

        # Create figure with controlled size and DPI
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot candlesticks manually
        for i, (idx, row) in enumerate(candles.iterrows()):
            color = 'green' if row['Close'] > row['Open'] else 'red'
            ax.plot([i, i], [row['Low'], row['High']], color='black', linewidth=1)
            height = abs(row['Close'] - row['Open'])
            bottom = min(row['Open'], row['Close'])
            rect = plt.Rectangle((i - 0.3, bottom), 0.6, height,
                                facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7)
            ax.add_patch(rect)

        # Define colors for Fibonacci levels (keeps visual distinction)
        fib_colors = ['red', 'orange', 'gold', 'green', 'blue', 'purple', 'brown']

        # Draw Fibonacci lines in the defined order (0% -> 100%)
        x_text = max(1, len(candles)) * 0.02
        for (level, price), color in zip(fib_levels, fib_colors):
            ax.axhline(y=price, linestyle="--", alpha=0.9, color=color, linewidth=1)
            # Use a small vertical offset for the text to reduce overlap
            ax.text(x_text, price, f"{level} ({price:.2f})",
                    va="bottom", ha="left", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor=color))

        # Axis limits and formatting
        ax.set_xlim(-1, len(candles))
        ax.set_ylim(candles[['Low', 'High']].min().min() * 0.98,
                    candles[['Low', 'High']].max().max() * 1.02)

        # Format x-axis with datetime labels (approx 10 ticks)
        step = max(1, len(candles) // 10)
        ax.set_xticks(range(0, len(candles), step))
        ax.set_xticklabels([candles.index[i].strftime('%Y-%m-%d') for i in range(0, len(candles), step)],
                        rotation=45, ha='right')

        # Titles and grid
        fig.suptitle('Candlestick Chart with Fibonacci Retracements', fontsize=14, y=0.95)
        ax.set_title(f"Trend: {trend.upper()} | Anchor from {swing_low:.2f} to {swing_high:.2f}",
                    fontsize=12, pad=10)
        ax.set_ylabel('Price')
        ax.grid(True, alpha=0.3)

        # Save settings
        dpi = 100
        try:
            fig.savefig("fibonacci_graph.png", format="png", dpi=dpi, pad_inches=0.1)
            print("Successfully saved fibonacci_graph.png")
        except Exception as e:
            print(f"Warning: Could not save local file: {e}")

        # Export to base64
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=dpi, pad_inches=0.1)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            print(f"First attempt failed: {e}. Trying with minimal settings...")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=72)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        plt.close(fig)

        return {
            "fibonacci_image": img_b64,
            "fibonacci_image_description": (
                f"Candlestick chart with Fibonacci retracement levels. "
                f"Trend detected as {trend.upper()}. Levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%) "
                f"are plotted between swing low {swing_low:.2f} and swing high {swing_high:.2f}. "
                f"Chart shows {len(candles)} data points."
            )
        }

    @staticmethod
    @tool
    def generate_trend_image(
        kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."]
    ) -> dict:
        """
        Generate a candlestick chart with trendlines from OHLCV data,
        save it locally as 'trend_graph.png', and return a base64-encoded image.

        Returns:
            dict: base64 image and description
        """
        data = pd.DataFrame(kline_data)
        candles = data.iloc[-50:].copy()

        candles["Datetime"] = pd.to_datetime(candles["Datetime"])
        candles.set_index("Datetime", inplace=True)

        # Trendline fit functions assumed to be defined outside this scope
        support_coefs_c, resist_coefs_c = fit_trendlines_single(candles['Close'])
        support_coefs, resist_coefs = fit_trendlines_high_low(candles['High'], candles['Low'], candles['Close'])

        # Trendline values
        support_line_c = support_coefs_c[0] * np.arange(len(candles)) + support_coefs_c[1]
        resist_line_c = resist_coefs_c[0] * np.arange(len(candles)) + resist_coefs_c[1]
        support_line = support_coefs[0] * np.arange(len(candles)) + support_coefs[1]
        resist_line = resist_coefs[0] * np.arange(len(candles)) + resist_coefs[1]

        # Convert to time-anchored coordinates
        s_seq = get_line_points(candles, support_line)
        r_seq = get_line_points(candles, resist_line)
        s_seq2 = get_line_points(candles, support_line_c)
        r_seq2 = get_line_points(candles, resist_line_c)

        s_segments = split_line_into_segments(s_seq)
        r_segments = split_line_into_segments(r_seq)
        s2_segments = split_line_into_segments(s_seq2)
        r2_segments = split_line_into_segments(r_seq2)

        all_segments = s_segments + r_segments + s2_segments + r2_segments
        colors = ['white'] * len(s_segments) + ['white'] * len(r_segments) + ['blue'] * len(s2_segments) + ['red'] * len(r2_segments)

        # Create addplot lines for close-based support/resistance
        apds = [
            mpf.make_addplot(support_line_c, color='blue', width=1, label="Close Support"),
            mpf.make_addplot(resist_line_c, color='red', width=1, label="Close Resistance")
        ]

        # Generate figure with legend and save locally
        fig, axlist = mpf.plot(
            candles,
            type='candle',
            style='charles',
            addplot=apds,
            alines=dict(alines=all_segments, colors=colors, linewidths=1),
            returnfig=True,
            figsize=(12, 6),
            block=False,
        )

        #save fig locally
        fig.savefig(
            "trend_graph.png",
            format="png",
            dpi=120,
            bbox_inches="tight",
            pad_inches=0.1
        )
        plt.close(fig) 

        # Add legend manually
        axlist[0].legend(loc='upper left')

        # Save to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)

        return {
            "trend_image": img_b64,
            "trend_image_description": "Trend-enhanced candlestick chart with support/resistance lines."
        }

    @staticmethod
    @tool
    def generate_kline_image(
        kline_data: Annotated[dict, "Dictionary containing OHLCV data with keys 'Datetime', 'Open', 'High', 'Low', 'Close'."],
    ) -> dict:
        """
        Generate a candlestick (K-line) chart from OHLCV data, save it locally, and return a base64-encoded image.

        Args:
            kline_data (dict): Dictionary with keys including 'Datetime', 'Open', 'High', 'Low', 'Close'.
            filename (str): Name of the file to save the image locally (default: 'kline_chart.png').

        Returns:
            dict: Dictionary containing base64-encoded image string and local file path.
        """

        df = pd.DataFrame(kline_data)
        # take recent 40
        df = df.tail(40)
        try:
            # df.index = pd.to_datetime(df["Datetime"])
            df.index = pd.to_datetime(df["Datetime"])

        except ValueError:
            print("ValueError at graph_util.py\n")



        # Save image locally
        fig, _ = mpf.plot(
            df[["Open", "High", "Low", "Close"]],
            type="candle",
            style="charles",
            figsize=(12, 6),
            returnfig=True,           
            block=False,             
            savefig=dict(             
                fname="kline_chart.png",
                dpi=120,
                bbox_inches="tight",
                pad_inches=0.1,
            ),
        )

        # ---------- Encode to base64 -----------------
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)                # release memory

        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "pattern_image": img_b64,
            "pattern_image_description": "Candlestick chart saved locally and returned as base64 string."
        }

    @staticmethod
    @tool
    def compute_rsi(
        kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        period: Annotated[int, "Lookback period for RSI calculation (default is 14)"] = 14
    ) -> dict:
        """
        Compute the Relative Strength Index (RSI) using TA-Lib.

        Args:
            data (dict): Dictionary containing at least a 'Close' key with a list of float values.
            period (int): Lookback period for RSI calculation (default is 14).

        Returns:
            dict: A dictionary with a single key 'rsi' mapping to a list of RSI values.
        """
        df = pd.DataFrame(kline_data)
        rsi = talib.RSI(df["Close"], timeperiod=period)
        return {"rsi": rsi.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    def compute_rsi_non_tool(
        kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        period: Annotated[int, "Lookback period for RSI calculation (default is 14)"] = 14
    ) -> dict:
        """
        Compute the Relative Strength Index (RSI) using TA-Lib.

        Args:
            data (dict): Dictionary containing at least a 'Close' key with a list of float values.
            period (int): Lookback period for RSI calculation (default is 14).

        Returns:
            dict: A dictionary with a single key 'rsi' mapping to a list of RSI values.
        """
        df = pd.DataFrame(kline_data)
        rsi = talib.RSI(df["Close"], timeperiod=period)
        return {"rsi": rsi.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    @tool
    def compute_macd(
        kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        fastperiod: Annotated[int, "Fast EMA period"] = 12,
        slowperiod: Annotated[int, "Slow EMA period"] = 26,
        signalperiod: Annotated[int, "Signal line EMA period"] = 9
    ) -> dict:
        """
        Compute the Moving Average Convergence Divergence (MACD) using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with list of float values.
            fastperiod (int): Fast EMA period.
            slowperiod (int): Slow EMA period.
            signalperiod (int): Signal line EMA period.

        Returns:
            dict: Dictionary containing 'macd', 'macd_signal', and 'macd_hist' as lists of values.
        """
        df = pd.DataFrame(kline_data)
        macd, macd_signal, macd_hist = talib.MACD(df["Close"], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return {
            "macd": macd.fillna(0).round(2).tolist()[-30:],
            "macd_signal": macd_signal.fillna(0).round(2).tolist()[-30:],
            "macd_hist": macd_hist.fillna(0).round(2).tolist()[-30:]
        }
    
    @staticmethod
    def compute_macd_non_tool(
        kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        fastperiod: Annotated[int, "Fast EMA period"] = 12,
        slowperiod: Annotated[int, "Slow EMA period"] = 26,
        signalperiod: Annotated[int, "Signal line EMA period"] = 9
    ) -> dict:
        """
        Compute the Moving Average Convergence Divergence (MACD) using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with list of float values.
            fastperiod (int): Fast EMA period.
            slowperiod (int): Slow EMA period.
            signalperiod (int): Signal line EMA period.

        Returns:
            dict: Dictionary containing 'macd', 'macd_signal', and 'macd_hist' as lists of values.
        """
        df = pd.DataFrame(kline_data)
        macd, macd_signal, macd_hist = talib.MACD(df["Close"], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
        return {
            "macd": macd.fillna(0).round(2).tolist()[-30:],
            "macd_signal": macd_signal.fillna(0).round(2).tolist()[-30:],
            "macd_hist": macd_hist.fillna(0).round(2).tolist()[-30:]
        }
    
    @staticmethod
    @tool
    def compute_stoch(kline_data: Annotated[dict, "Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values."]
    ) -> dict:
        """
        Compute the Stochastic Oscillator %K and %D using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values.

        Returns:
            dict: A dictionary with keys 'stoch_k' and 'stoch_d',
                each mapping to a list representing %K and %D values.
        """
        df = pd.DataFrame(kline_data)
        stoch_k, stoch_d = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=14, slowk_period=3, slowd_period=3)
        return {
            "stoch_k": stoch_k.fillna(0).round(2).tolist()[-30:],
            "stoch_d": stoch_d.fillna(0).round(2).tolist()[-30:]
        }
    
    @staticmethod
    def compute_stoch_non_tool(kline_data: Annotated[dict, "Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values."]
    ) -> dict:
        """
        Compute the Stochastic Oscillator %K and %D using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys, each mapping to lists of float values.

        Returns:
            dict: A dictionary with keys 'stoch_k' and 'stoch_d',
                each mapping to a list representing %K and %D values.
        """
        df = pd.DataFrame(kline_data)
        stoch_k, stoch_d = talib.STOCH(df["High"], df["Low"], df["Close"], fastk_period=14, slowk_period=3, slowd_period=3)
        return {
            "stoch_k": stoch_k.fillna(0).round(2).tolist()[-30:],
            "stoch_d": stoch_d.fillna(0).round(2).tolist()[-30:]
        }
    
    @staticmethod
    @tool
    def compute_roc(kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        period: Annotated[int, "Number of periods over which to calculate ROC (default is 10)"] = 10
    ) -> dict:
        """
        Compute the Rate of Change (ROC) indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with a list of float values.
            period (int): Number of periods over which to calculate ROC (default is 10).

        Returns:
            dict: A dictionary with a single key 'roc' mapping to a list of ROC values.
        """

        df = pd.DataFrame(kline_data)
        roc = talib.ROC(df["Close"], timeperiod=period)
        return {"roc": roc.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    def compute_roc_non_tool(kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        period: Annotated[int, "Number of periods over which to calculate ROC (default is 10)"] = 10
    ) -> dict:
        """
        Compute the Rate of Change (ROC) indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with a list of float values.
            period (int): Number of periods over which to calculate ROC (default is 10).

        Returns:
            dict: A dictionary with a single key 'roc' mapping to a list of ROC values.
        """

        df = pd.DataFrame(kline_data)
        roc = talib.ROC(df["Close"], timeperiod=period)
        return {"roc": roc.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    @tool
    def compute_willr(
        kline_data: Annotated[dict, "Dictionary with 'High', 'Low', and 'Close' keys containing float lists."],
        period: Annotated[int, "Lookback period for Williams %R"] = 14
    ) -> dict:
        """
        Compute the Williams %R indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys.
            period (int): Lookback period for Williams %R calculation.

        Returns:
            dict: Dictionary with key 'willr' mapping to the list of Williams %R values.
        """
        # print("-------------------------CALLED COMPUTE WILLR--------------------------\n")
        df = pd.DataFrame(kline_data)
        willr = talib.WILLR(df["High"], df["Low"], df["Close"], timeperiod=period)
        return {"willr": willr.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    def compute_willr_non_tool(
        kline_data: Annotated[dict, "Dictionary with 'High', 'Low', and 'Close' keys containing float lists."],
        period: Annotated[int, "Lookback period for Williams %R"] = 14
    ) -> dict:
        """
        Compute the Williams %R indicator using TA-Lib.

        Args:
            kline_data (dict): Dictionary with 'High', 'Low', and 'Close' keys.
            period (int): Lookback period for Williams %R calculation.

        Returns:
            dict: Dictionary with key 'willr' mapping to the list of Williams %R values.
        """
        # print("-------------------------CALLED COMPUTE WILLR--------------------------\n")
        df = pd.DataFrame(kline_data)
        willr = talib.WILLR(df["High"], df["Low"], df["Close"], timeperiod=period)
        return {"willr": willr.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    @tool
    def compute_anchored_vwap(
        kline_data: Annotated[dict, "Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys containing lists."],
        start_time: Annotated[Optional[str], "Starting datetime in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' to anchor VWAP calculation"] = None
    ) -> dict: 
        """
        Compute the Anchored VWAP starting from a specific datetime.
        Note: TA-Lib does not have a built-in VWAP function, so this uses manual calculation.
        
        Args:
            kline_data (dict): Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys.
            start_time (str): Starting datetime in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        
        Returns:
            dict: Dictionary with key 'anchored_vwap' mapping to the list of VWAP values.
        """
        df = pd.DataFrame(kline_data)
        
        # Convert Datetime column to pandas datetime
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Find the starting index based on datetime
        try:
            start_datetime = df['Datetime'].iloc[0] if start_time is None else pd.to_datetime(start_time)
            # Find the closest datetime index
            start_idx = df[df['Datetime'] >= start_datetime].index[0]
        except (ValueError, IndexError):
            # If parsing fails or no matching date found, start from beginning
            start_idx = 0
        
        # Calculate typical price for each period
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        
        # Initialize arrays
        anchored_vwap = np.full(len(df), np.nan)
        
        # Calculate cumulative values starting from start_idx
        cumulative_pv = 0
        cumulative_volume = 0
        
        for i in range(start_idx, len(df)):
            # Add current period's price * volume and volume
            cumulative_pv += typical_price.iloc[i] * df["Volume"].iloc[i]
            cumulative_volume += df["Volume"].iloc[i]
            
            # Calculate VWAP
            if cumulative_volume > 0:
                anchored_vwap[i] = cumulative_pv / cumulative_volume
        
        # Convert to pandas Series for easier handling
        anchored_vwap_series = pd.Series(anchored_vwap)
        
        return {"anchored_vwap": anchored_vwap_series.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    def compute_anchored_vwap_non_tool(
        kline_data: Annotated[dict, "Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys containing lists."],
        start_time: Annotated[Optional[str], "Starting datetime in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' to anchor VWAP calculation"] = None
    ) -> dict: 
        """
        Compute the Anchored VWAP starting from a specific datetime.
        Note: TA-Lib does not have a built-in VWAP function, so this uses manual calculation.
        
        Args:
            kline_data (dict): Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys.
            start_time (str): Starting datetime in format 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.
        
        Returns:
            dict: Dictionary with key 'anchored_vwap' mapping to the list of VWAP values.
        """
        df = pd.DataFrame(kline_data)
        
        # Convert Datetime column to pandas datetime
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Find the starting index based on datetime
        try:
            start_datetime = df['Datetime'].iloc[0] if start_time is None else pd.to_datetime(start_time)
            # Find the closest datetime index
            start_idx = df[df['Datetime'] >= start_datetime].index[0]
        except (ValueError, IndexError):
            # If parsing fails or no matching date found, start from beginning
            start_idx = 0
        
        # Calculate typical price for each period
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        
        # Initialize arrays
        anchored_vwap = np.full(len(df), np.nan)
        
        # Calculate cumulative values starting from start_idx
        cumulative_pv = 0
        cumulative_volume = 0
        
        for i in range(start_idx, len(df)):
            # Add current period's price * volume and volume
            cumulative_pv += typical_price.iloc[i] * df["Volume"].iloc[i]
            cumulative_volume += df["Volume"].iloc[i]
            
            # Calculate VWAP
            if cumulative_volume > 0:
                anchored_vwap[i] = cumulative_pv / cumulative_volume
        
        # Convert to pandas Series for easier handling
        anchored_vwap_series = pd.Series(anchored_vwap)
        
        return {"anchored_vwap": anchored_vwap_series.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    @tool
    def compute_vwap(
        kline_data: Annotated[dict, "Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys containing lists."]
    ) -> dict: 
        """
        Compute the Volume Weighted Average Price (VWAP) using cumulative calculation.
        VWAP represents the average price weighted by volume from the start of the trading period.
        Note: TA-Lib does not have a built-in VWAP function, so this uses manual calculation.
        
        Args:
            kline_data (dict): Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys.
        
        Returns:
            dict: Dictionary with key 'vwap' mapping to the list of VWAP values.
        """
        df = pd.DataFrame(kline_data)
        
        # Calculate typical price for each period
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        
        # Calculate price * volume
        price_volume = typical_price * df["Volume"]
        
        # Calculate cumulative VWAP
        cumulative_pv = price_volume.cumsum()
        cumulative_volume = df["Volume"].cumsum()
        
        # Calculate VWAP (avoid division by zero)
        vwap = cumulative_pv / cumulative_volume.replace(0, np.nan)
        
        return {"vwap": vwap.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    def compute_vwap_non_tool(
        kline_data: Annotated[dict, "Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys containing lists."]
    ) -> dict: 
        """
        Compute the Volume Weighted Average Price (VWAP) using cumulative calculation.
        VWAP represents the average price weighted by volume from the start of the trading period.
        Note: TA-Lib does not have a built-in VWAP function, so this uses manual calculation.
        
        Args:
            kline_data (dict): Dictionary with 'High', 'Low', 'Close', 'Volume', and 'Datetime' keys.
        
        Returns:
            dict: Dictionary with key 'vwap' mapping to the list of VWAP values.
        """
        df = pd.DataFrame(kline_data)
        
        # Calculate typical price for each period
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        
        # Calculate price * volume
        price_volume = typical_price * df["Volume"]
        
        # Calculate cumulative VWAP
        cumulative_pv = price_volume.cumsum()
        cumulative_volume = df["Volume"].cumsum()
        
        # Calculate VWAP (avoid division by zero)
        vwap = cumulative_pv / cumulative_volume.replace(0, np.nan)
        
        return {"vwap": vwap.fillna(0).round(2).tolist()[-30:]}

    @staticmethod
    @tool
    def compute_sma(
        kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        short_term_period: Annotated[int, "Number of periods for the short term Simple Moving Average (default is 20)"] = 20,
        long_term_period: Annotated[int, "Number of periods for the long term Simple Moving Average (default is 40)"] = 40,

    ) -> dict:
        """
        Compute the Simple Moving Average (SMA) using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with a list of float values.
            period (int): Number of periods for the Simple Moving Average (default is 20).

        Returns:
            dict: A dictionary with a single key 'sma' mapping to a list of SMA values.
        """
        df = pd.DataFrame(kline_data)
        st_sma = talib.SMA(df["Close"], timeperiod=short_term_period)
        lt_sma = talib.SMA(df["Close"], timeperiod=long_term_period)
        return {"short_term_sma": st_sma.fillna(0).round(2).tolist()[-30:],
                "long_term_sma": lt_sma.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    def compute_sma_non_tool(
        kline_data: Annotated[dict, "Dictionary with a 'Close' key containing a list of float closing prices."],
        short_term_period: Annotated[int, "Number of periods for the short term Simple Moving Average (default is 20)"] = 20,
        long_term_period: Annotated[int, "Number of periods for the long term Simple Moving Average (default is 40)"] = 40,

    ) -> dict:
        """
        Compute the Simple Moving Average (SMA) using TA-Lib.

        Args:
            kline_data (dict): Dictionary containing a 'Close' key with a list of float values.
            period (int): Number of periods for the Simple Moving Average (default is 20).

        Returns:
            dict: A dictionary with a single key 'sma' mapping to a list of SMA values.
        """
        df = pd.DataFrame(kline_data)
        st_sma = talib.SMA(df["Close"], timeperiod=short_term_period)
        lt_sma = talib.SMA(df["Close"], timeperiod=long_term_period)
        return {"short_term_sma": st_sma.fillna(0).round(2).tolist()[-30:],
                "long_term_sma": lt_sma.fillna(0).round(2).tolist()[-30:]}
    
    @staticmethod
    @tool
    def compute_heiken_ashi(
        kline_data: Annotated[dict, "Dictionary with 'Open', 'High', 'Low', and 'Close' keys containing float lists."]
    ) -> dict:
        """
        Compute Heiken-Ashi candlestick values from traditional OHLC data.
        
        Args:
            kline_data (dict): Dictionary with 'Open', 'High', 'Low', and 'Close' keys.
        
        Returns:
            dict: Dictionary with keys 'ha_open', 'ha_high', 'ha_low', 'ha_close',
                each mapping to a list of Heiken-Ashi values.
        """
        df = pd.DataFrame(kline_data)
        
        # Initialize arrays for Heiken-Ashi values
        ha_open = np.zeros(len(df))
        ha_high = np.zeros(len(df))
        ha_low = np.zeros(len(df))
        ha_close = np.zeros(len(df))
        
        for i in range(len(df)):
            # Calculate Heiken-Ashi Close: (O + H + L + C) / 4
            ha_close[i] = (df["Open"].iloc[i] + df["High"].iloc[i] + 
                        df["Low"].iloc[i] + df["Close"].iloc[i]) / 4
            
            if i == 0:
                # First candle initialization
                # HA Open = (Open + Close) / 2
                ha_open[i] = (df["Open"].iloc[i] + df["Close"].iloc[i]) / 2
                # HA High = Original High
                ha_high[i] = df["High"].iloc[i]
                # HA Low = Original Low  
                ha_low[i] = df["Low"].iloc[i]
            else:
                # Subsequent candles
                # HA Open = (Previous HA Open + Previous HA Close) / 2
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
                
                # HA High = MAX(Original High, HA Open, HA Close)
                ha_high[i] = max(df["High"].iloc[i], ha_open[i], ha_close[i])
                
                # HA Low = MIN(Original Low, HA Open, HA Close)
                ha_low[i] = min(df["Low"].iloc[i], ha_open[i], ha_close[i])
        
        return {
            "ha_open": np.round(ha_open, 2).tolist()[-30:],
            "ha_high": np.round(ha_high, 2).tolist()[-30:],
            "ha_low": np.round(ha_low, 2).tolist()[-30:],
            "ha_close": np.round(ha_close, 2).tolist()[-30:]
        }
    
    @staticmethod
    def compute_heiken_ashi_non_tool(
        kline_data: Annotated[dict, "Dictionary with 'Open', 'High', 'Low', and 'Close' keys containing float lists."]
    ) -> dict:
        """
        Compute Heiken-Ashi candlestick values from traditional OHLC data.
        
        Args:
            kline_data (dict): Dictionary with 'Open', 'High', 'Low', and 'Close' keys.
        
        Returns:
            dict: Dictionary with keys 'ha_open', 'ha_high', 'ha_low', 'ha_close',
                each mapping to a list of Heiken-Ashi values.
        """
        df = pd.DataFrame(kline_data)
        
        # Initialize arrays for Heiken-Ashi values
        ha_open = np.zeros(len(df))
        ha_high = np.zeros(len(df))
        ha_low = np.zeros(len(df))
        ha_close = np.zeros(len(df))
        
        for i in range(len(df)):
            # Calculate Heiken-Ashi Close: (O + H + L + C) / 4
            ha_close[i] = (df["Open"].iloc[i] + df["High"].iloc[i] + 
                        df["Low"].iloc[i] + df["Close"].iloc[i]) / 4
            
            if i == 0:
                # First candle initialization
                # HA Open = (Open + Close) / 2
                ha_open[i] = (df["Open"].iloc[i] + df["Close"].iloc[i]) / 2
                # HA High = Original High
                ha_high[i] = df["High"].iloc[i]
                # HA Low = Original Low  
                ha_low[i] = df["Low"].iloc[i]
            else:
                # Subsequent candles
                # HA Open = (Previous HA Open + Previous HA Close) / 2
                ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
                
                # HA High = MAX(Original High, HA Open, HA Close)
                ha_high[i] = max(df["High"].iloc[i], ha_open[i], ha_close[i])
                
                # HA Low = MIN(Original Low, HA Open, HA Close)
                ha_low[i] = min(df["Low"].iloc[i], ha_open[i], ha_close[i])
        
        return {
            "ha_open": np.round(ha_open, 2).tolist()[-30:],
            "ha_high": np.round(ha_high, 2).tolist()[-30:],
            "ha_low": np.round(ha_low, 2).tolist()[-30:],
            "ha_close": np.round(ha_close, 2).tolist()[-30:]
        }
    
    @staticmethod
    # @tool
    def standard_normalize(vec):
        """
        Normalize a vector using standard normalization (z-score).
        
        Args:
            vec: Input vector to normalize
            
        Returns:
            Normalized vector with mean=0 and std=1
        """
        arr = np.array(vec, dtype=float)
        if not np.isfinite(arr).all():
            raise ValueError("Indicator vector contains non-finite values")
        std = arr.std()
        return np.zeros_like(arr) if std < 1e-12 else (arr - arr.mean()) / std
    
    @staticmethod
    def simulate_trade_outcome(kline_data: dict, decision_index: int, decision: str, horizon: int = 3) -> dict:
        """
        Simulate the outcome of a trade decision.

        Args:
            kline_data (dict): Dict containing OHLCV arrays, including 'Close'.
            decision_index (int): Index of the candle where decision is made.
            decision (str): "LONG" or "SHORT".
            horizon (int): How many candles ahead to evaluate.

        Returns:
            dict: outcome, pnl, entry_price, exit_price
        """
        closes = kline_data["Close"]
        if horizon < 1 or decision_index < 0 or decision_index + horizon >= len(closes):
            return {"outcome": "INVALID", "pnl": 0, "entry_price": None, "exit_price": None}

        entry_price = closes[decision_index]
        end_index = min(decision_index + horizon, len(closes) - 1)
        exit_price = closes[end_index]

        # Simple P&L calculation
        if decision.upper() == "LONG":
            pnl = exit_price - entry_price
        elif decision.upper() == "SHORT":
            pnl = entry_price - exit_price
        else:
            return {"outcome": "UNKNOWN", "pnl": 0, "entry_price": entry_price, "exit_price": exit_price}

        outcome = "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "NEUTRAL"

        return {
            "outcome": outcome,
            "pnl": pnl,
            "entry_price": entry_price,
            "exit_price": exit_price
        }


    @staticmethod
    def compute_normalized_ohlcv_vector(kline_data: dict) -> np.ndarray:
        """
        Compute a concatenated vector containing only normalized OHLCV data.
        
        Args:
            kline_data (dict): Dictionary with keys 'Open', 'High', 'Low', 'Close', 'Volume'.
        
        Returns:
            np.ndarray: Concatenated normalized OHLCV vector.
        """
        
        # Normalize each OHLCV series
        open_vector = TechnicalTools.standard_normalize(kline_data['Open'][-30:])
        high_vector = TechnicalTools.standard_normalize(kline_data['High'][-30:])
        low_vector = TechnicalTools.standard_normalize(kline_data['Low'][-30:])
        close_vector = TechnicalTools.standard_normalize(kline_data['Close'][-30:])
        volume_vector = TechnicalTools.standard_normalize(kline_data['Volume'][-30:])
        
        # Concatenate into one long vector
        ohlcv_vector = np.concatenate([
            open_vector,
            high_vector,
            low_vector,
            close_vector,
            volume_vector
        ])
        
        return ohlcv_vector

    @staticmethod
    # @tool
    def compute_all_trading_indicators(kline_data: dict) -> np.ndarray:
        """
        Compute all trading indicators from a CSV file and concatenate them into a single vector.
        
        Args:
            input_file (str): Path to the CSV file containing OHLCV data with columns:
                            'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'
        
        Returns:
            list: Concatenated vector containing all trading indicators in the following order:
                [anchored_vwap, ha_close, macd, macd_signal, macd_hist, roc, rsi, 
                short_term_sma, stoch_k, stoch_d, willr]
        """
        
        # Compute all indicators
        # Anchored VWAP (using first datetime as anchor)
        avwap_dict = TechnicalTools.compute_anchored_vwap_non_tool(kline_data, kline_data['Datetime'][0])
        avwap_vector = TechnicalTools.standard_normalize(avwap_dict["anchored_vwap"])

        # VWAP
        # vwap_dict = TechnicalTools.compute_vwap_non_tool(kline_data)
        # vwap_vector = TechnicalTools.standard_normalize(vwap_dict["vwap"])
        
        # Heiken-Ashi
        ha_dict = TechnicalTools.compute_heiken_ashi_non_tool(kline_data)
        ha_close_vector = TechnicalTools.standard_normalize(ha_dict["ha_close"])
        ha_open_vector = TechnicalTools.standard_normalize(ha_dict["ha_open"])
        ha_high_vector = TechnicalTools.standard_normalize(ha_dict["ha_high"])
        ha_low_vector = TechnicalTools.standard_normalize(ha_dict["ha_low"])

        
        # MACD
        macd_dict = TechnicalTools.compute_macd_non_tool(kline_data)
        macd_vector = TechnicalTools.standard_normalize(macd_dict["macd"])
        macd_signal_vector = TechnicalTools.standard_normalize(macd_dict["macd_signal"])
        macd_hist_vector = TechnicalTools.standard_normalize(macd_dict["macd_hist"])
        
        # Rate of Change
        roc_dict = TechnicalTools.compute_roc_non_tool(kline_data)
        roc_vector = TechnicalTools.standard_normalize(roc_dict["roc"])
        
        # RSI
        rsi_dict = TechnicalTools.compute_rsi_non_tool(kline_data)
        rsi_vector = TechnicalTools.standard_normalize(rsi_dict["rsi"])
        
        # Simple Moving Average
        sma_dict = TechnicalTools.compute_sma_non_tool(kline_data)
        st_sma_vector = TechnicalTools.standard_normalize(sma_dict["short_term_sma"])
        lt_sma_vector = TechnicalTools.standard_normalize(sma_dict["long_term_sma"])
        
        # Stochastic Oscillator
        stoch_dict = TechnicalTools.compute_stoch_non_tool(kline_data)
        stoch_k_vector = TechnicalTools.standard_normalize(stoch_dict['stoch_k'])  # Fixed typo: was 'stock_k'
        stoch_d_vector = TechnicalTools.standard_normalize(stoch_dict['stoch_d'])  # Fixed typo: was 'stock_d'
        
        # Williams %R
        willr_dict = TechnicalTools.compute_willr_non_tool(kline_data)
        willr_vector = TechnicalTools.standard_normalize(willr_dict['willr'])
        
        # Concatenate all vectors into a single long vector
        combined_vector = np.concatenate([
            avwap_vector,
            ha_close_vector,
            ha_high_vector,
            ha_low_vector,
            ha_open_vector,
            macd_vector,
            macd_signal_vector,
            macd_hist_vector,
            roc_vector,
            rsi_vector,
            st_sma_vector,
            lt_sma_vector,
            stoch_k_vector,
            stoch_d_vector,
            willr_vector
        ])
        
        return combined_vector
    
    @staticmethod
    @tool
    def compute_market_similarities(folder_path: str) -> dict:
        """
        Compute cosine similarity between all market moments in a folder.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            
        Returns:
            dict: {filename: {other_filename: similarity_score}} with top 10 most similar for each file
        """
        # Get all CSV files
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        csv_files.sort()
        
        # Get filenames without path
        filenames = [os.path.basename(f) for f in csv_files]
        
        # Compute indicator vectors for all files
        vectors = []
        valid_files = []
        
        for file_path, filename in zip(csv_files, filenames):
            # Read the CSV
            df = pd.read_csv(file_path)
            
            # Make sure the column names match exactly
            # Example: if they are lowercase in the CSV, rename them
            df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
            
            # Convert to dict format required by the indicator functions
            kline_data = {
                "Datetime": df["Datetime"].tolist(),
                "Open": df["Open"].tolist(),
                "High": df["High"].tolist(),
                "Low": df["Low"].tolist(),
                "Close": df["Close"].tolist(),
                "Volume": df["Volume"].tolist()
            }

            try:
                vector = TechnicalTools.compute_all_trading_indicators(kline_data)
                vectors.append(vector)
                valid_files.append(filename)
            except:
                print(f"Skipped {filename} - error processing")
                continue
        
        # Convert to numpy array and compute cosine similarity
        vector_matrix = np.array(vectors)
        similarity_matrix = cosine_similarity(vector_matrix)
        
        # Convert to dictionary format with top 10 similarities only
        result = {}
        for i, filename in enumerate(valid_files):
            # Get all similarities for this file
            similarities = {}
            for j, other_filename in enumerate(valid_files):
                if other_filename != filename:  # Exclude self
                    similarities[other_filename] = similarity_matrix[i][j]
            
            # Keep only top 10 most similar
            top_5 = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5])
            result[filename] = top_5
        
        return result
    
    @staticmethod
    @tool
    def compute_market_price_similarities(folder_path: str) -> dict:
        """
        Compute cosine similarity between all market moments in a folder.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            
        Returns:
            dict: {filename: {other_filename: similarity_score}} with top 10 most similar for each file
        """
        # Get all CSV files
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        csv_files.sort()
        
        # Get filenames without path
        filenames = [os.path.basename(f) for f in csv_files]
        
        # Compute indicator vectors for all files
        vectors = []
        valid_files = []
        
        for file_path, filename in zip(csv_files, filenames):
            try:
                df = pd.read_csv(file_path)
                # Keep only latest 30 time periods
                df = df.tail(30).reset_index(drop=True)
                
                # Make sure the column names match exactly
                # Example: if they are lowercase in the CSV, rename them
                df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
    
                close_vector = df["Close"].tolist()
                close_vector = TechnicalTools.standard_normalize(close_vector)
                high_vector = df["High"].tolist()
                high_vector = TechnicalTools.standard_normalize(high_vector)
                low_vector = df["Low"].tolist()
                low_vector = TechnicalTools.standard_normalize(low_vector)
                open_vector = df["Open"].tolist()
                open_vector = TechnicalTools.standard_normalize(open_vector)
                volume_vector = df["Volume"].tolist()
                volume_vector = TechnicalTools.standard_normalize(volume_vector)

                vector = (close_vector + high_vector + low_vector + open_vector + volume_vector)

                vectors.append(vector)
                valid_files.append(filename)
            except:
                print(f"Skipped {filename} - error processing")
                continue
        
        # Convert to numpy array and compute cosine similarity
        vector_matrix = np.array(vectors)
        similarity_matrix = cosine_similarity(vector_matrix)
        
        # Convert to dictionary format with top 10 similarities only
        result = {}
        for i, filename in enumerate(valid_files):
            # Get all similarities for this file
            similarities = {}
            for j, other_filename in enumerate(valid_files):
                if other_filename != filename:  # Exclude self
                    similarities[other_filename] = similarity_matrix[i][j]
            
            # Keep only top 10 most similar
            top_5 = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5])
            result[filename] = top_5
        
        return result
    
    @staticmethod
    @tool
    def check_alignment(indicator_similarities: dict, price_similarities: dict) -> dict:
        """
        Check how many files overlap in top-10 lists between the two methods.
        Returns: {filename: intersection_count}
        """
        alignment_results = {}
        
        common_files = set(indicator_similarities.keys()) & set(price_similarities.keys())
        
        for filename in common_files:
            indicator_top5 = set(indicator_similarities[filename].keys())
            price_top5 = set(price_similarities[filename].keys())
            intersection_count = len(indicator_top5 & price_top5)
            alignment_results[filename] = intersection_count
        
        # Print summary
        avg_intersection = sum(alignment_results.values()) / len(alignment_results)
        print(f"Average overlap: {avg_intersection:.2f} out of 5")
        print(f"Total files: {len(alignment_results)}")
        
        return alignment_results
    
    @staticmethod
    @tool
    def generate_all_kline_plots(folder_path: str, output_folder: str = "experiment_images/btc_img"):
        """
        Generate K-line plots for all CSV files in a folder and save them as images.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            output_folder (str): Folder to save the generated images
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all CSV files
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        csv_files.sort()
        
        print(f"Found {len(csv_files)} CSV files")
        print(f"Saving images to: {output_folder}")
        
        for i, file_path in enumerate(csv_files):
            filename = os.path.basename(file_path)
            filename_no_ext = os.path.splitext(filename)[0]
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
                
                # Convert to required dictionary format
                kline_data = {
                    "Datetime": df["Datetime"].tolist(),
                    "Open": df["Open"].tolist(),
                    "High": df["High"].tolist(),
                    "Low": df["Low"].tolist(),
                    "Close": df["Close"].tolist()
                }
                
                # Take last 40 records and prepare for plotting
                df_plot = pd.DataFrame(kline_data)
                df_plot.index = pd.to_datetime(df_plot["Datetime"], format="%Y-%m-%d %H:%M:%S")
                
                # Generate plot and save to specific location
                output_path = os.path.join(output_folder, f"{filename_no_ext}_kline.png")
                
                fig, _ = mpf.plot(
                    df_plot[["Open", "High", "Low", "Close"]],
                    type="candle",
                    style="charles",
                    figsize=(12, 6),
                    returnfig=True,
                    block=False,
                    title=f"K-line Chart: {filename_no_ext}",
                    savefig=dict(
                        fname=output_path,
                        dpi=300,
                        bbox_inches="tight",
                        pad_inches=0.1,
                    ),
                )
                
                plt.close(fig)  # Release memory
                print(f"({i+1}/{len(csv_files)}) Saved: {output_path}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"\nCompleted! All images saved to '{output_folder}' folder.")

    @staticmethod
    # @tool
    def compute_market_all_similarities(folder_path: str) -> dict:
        """
        Compute cosine similarity between all market moments in a folder.
        
        Args:
            folder_path (str): Path to folder containing CSV files
            
        Returns:
            dict: {filename: {other_filename: similarity_score}} with top 10 most similar for each file
        """
        # Get all CSV files
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        csv_files.sort()
        
        # Get filenames without path
        filenames = [os.path.basename(f) for f in csv_files]
        
        # Compute indicator vectors for all files
        vectors = []
        valid_files = []
        
        for file_path, filename in zip(csv_files, filenames):
            try:
                df = pd.read_csv(file_path)
                # Keep only latest 30 time periods
                df = df.tail(30).reset_index(drop=True)
                
                # Make sure the column names match exactly
                # Example: if they are lowercase in the CSV, rename them
                df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
    
                close_vector = df["Close"].tolist()
                close_vector = TechnicalTools.standard_normalize(close_vector)
                high_vector = df["High"].tolist()
                high_vector = TechnicalTools.standard_normalize(high_vector)
                low_vector = df["Low"].tolist()
                low_vector = TechnicalTools.standard_normalize(low_vector)
                open_vector = df["Open"].tolist()
                open_vector = TechnicalTools.standard_normalize(open_vector)
                volume_vector = df["Volume"].tolist()
                volume_vector = TechnicalTools.standard_normalize(volume_vector)

                vector = np.concatenate([close_vector, high_vector, low_vector, open_vector, volume_vector])

                # Read the CSV
                df = pd.read_csv(file_path)
                
                # Make sure the column names match exactly
                # Example: if they are lowercase in the CSV, rename them
                df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
                
                # Convert to dict format required by the indicator functions
                kline_data = {
                    "Datetime": df["Datetime"].tolist(),
                    "Open": df["Open"].tolist(),
                    "High": df["High"].tolist(),
                    "Low": df["Low"].tolist(),
                    "Close": df["Close"].tolist(),
                    "Volume": df["Volume"].tolist()
                }

                indicator_vector = TechnicalTools.compute_all_trading_indicators(kline_data)

                vector = np.concatenate([vector, indicator_vector])

                vectors.append(vector)
                valid_files.append(filename)
            except:
                print(f"Skipped {filename} - error processing")
                continue
        
        # Convert to numpy array and compute cosine similarity
        vector_matrix = np.array(vectors)
        similarity_matrix = cosine_similarity(vector_matrix)
        
        # Convert to dictionary format with top 10 similarities only
        result = {}
        for i, filename in enumerate(valid_files):
            # Get all similarities for this file
            similarities = {}
            for j, other_filename in enumerate(valid_files):
                if other_filename != filename:  # Exclude self
                    similarities[other_filename] = similarity_matrix[i][j]
            
            # Keep only top 10 most similar
            top_5 = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5])
            result[filename] = top_5
        
        return result

