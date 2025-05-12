import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_stock_data(start_date, end_date, initial_price=100, volatility=0.2, trend=0.001):
    """
    Generates synthetic stock price data with a random walk and an upward trend.

    Args:
        start_date (str): The start date for the data (e.g., '2023-01-01').
        end_date (str): The end date for the data (e.g., '2023-12-31').
        initial_price (float): The initial stock price.
        volatility (float): The volatility of the stock price (annualized).
        trend (float): The daily trend of the stock price.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Date' and 'StockPrice'.
    """
    dates = pd.date_range(start=start_date, end=end_date)
    num_days = len(dates)
    daily_volatility = volatility / np.sqrt(252)  # Assume 252 trading days in a year
    stock_prices = [initial_price]

    for _ in range(num_days - 1):
        daily_return = trend + np.random.normal(0, daily_volatility)
        new_price = stock_prices[-1] * np.exp(daily_return)
        stock_prices.append(new_price)

    df = pd.DataFrame({'Date': dates, 'StockPrice': stock_prices})
    return df

def generate_synthetic_option_data(stock_data, option_type='call', num_options=100):
    """
    Generates synthetic option data based on the given stock data.  Uses Black-Scholes for option pricing.

    Args:
        stock_data (pd.DataFrame): DataFrame containing stock price data with columns 'Date' and 'StockPrice'.
        option_type (str):  'call' or 'put'.
        num_options: number of options to generate

    Returns:
        pd.DataFrame: A DataFrame with columns 'Date', 'OptionType', 'StrikePrice', 'ExpirationDate', 'OptionPrice'.
    """

    option_data = []
    for _ in range(num_options):
        # Randomly sample a date from the stock data
        option_date_index = np.random.randint(0, len(stock_data))
        option_date = stock_data['Date'][option_date_index]
        stock_price = stock_data['StockPrice'][option_date_index]

        # Generate strike price (around the stock price)
        strike_price = stock_price * np.random.uniform(0.8, 1.2)
        # Generate expiration date (between 1 week and 1 year)
        expiration_date = option_date + timedelta(days=np.random.randint(7, 365))

        # Ensure the expiration date is within the stock data date range
        if expiration_date > stock_data['Date'].max():
            expiration_date = stock_data['Date'].max()

        # Calculate time to expiration in years
        time_to_expiration = (expiration_date - option_date).days / 365.0
        if time_to_expiration <= 0:
            continue  # Skip if time to expiration is zero or negative

        # Use Black-Scholes to calculate option price.
        interest_rate = 0.05  # Assume a 5% risk-free interest rate
        volatility = 0.2  # Assume 20% volatility (you might want to vary this)
        option_price = calculate_black_scholes_price(stock_price, strike_price, time_to_expiration, interest_rate, volatility, option_type)

        option_data.append({
            'Date': option_date,
            'OptionType': option_type,
            'StrikePrice': strike_price,
            'ExpirationDate': expiration_date,
            'OptionPrice': option_price,
            'StockPrice': stock_price  # Include the corresponding stock price
        })
    return pd.DataFrame(option_data)

def calculate_black_scholes_price(stock_price, strike_price, time_to_expiration, interest_rate, volatility, option_type='call'):
    """
    Calculates the Black-Scholes option price.

    Args:
        stock_price (float): Current stock price.
        strike_price (float): Strike price of the option.
        time_to_expiration (float): Time to expiration in years.
        interest_rate (float): Risk-free interest rate.
        volatility (float): Volatility of the underlying asset.
        option_type (str): 'call' or 'put'.

    Returns:
        float: The Black-Scholes option price.
    """
    from scipy.stats import norm
    d1 = (np.log(stock_price / strike_price) + (interest_rate + 0.5 * volatility ** 2) * time_to_expiration) / (volatility * np.sqrt(time_to_expiration))
    d2 = d1 - volatility * np.sqrt(time_to_expiration)

    if option_type == 'call':
        option_price = stock_price * norm.cdf(d1) - strike_price * np.exp(-interest_rate * time_to_expiration) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = strike_price * np.exp(-interest_rate * time_to_expiration) * norm.cdf(-d2) - stock_price * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return option_price

if __name__ == "__main__":
    # Generate synthetic stock data
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    stock_data_df = generate_synthetic_stock_data(start_date, end_date)

    # Generate synthetic call option data
    call_option_data_df = generate_synthetic_option_data(stock_data_df, option_type='call', num_options=150)

    # Generate synthetic put option data
    put_option_data_df = generate_synthetic_option_data(stock_data_df, option_type='put', num_options=150)

    # Combine the call and put option data
    option_data_df = pd.concat([call_option_data_df, put_option_data_df], ignore_index=True)


    # Save the data to CSV files
    stock_data_df.to_csv('synthetic_stock_data.csv', index=False)
    option_data_df.to_csv('synthetic_option_data.csv', index=False)

    print("Synthetic stock and option data generated and saved to 'synthetic_stock_data.csv' and 'synthetic_option_data.csv'")
