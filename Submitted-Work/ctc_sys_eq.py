import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as plt
from collections import defaultdict
import math
import matplotlib.pyplot as plt


def get_data(portfolio, start_date, end_date):
    # Initialize an empty DataFrame
    data = pd.DataFrame()

    # Create an empty list for columns
    columns = []

    # Loop through each country in the portfolio
    for country, tickers in portfolio.items():
        # Download data for each ticker
        df = yf.download(tickers, start=start_date, end=end_date)

        # Extract the columns representing stock attributes (e.g., 'Adj Close', 'Volume', etc.)
        stock_attributes = df.columns.levels[0]

        # Extend the list of columns with country and attribute prefix
        country_columns = [(country, attr, ticker) for ticker in tickers for attr in stock_attributes]
        columns.extend(country_columns)

        # Concatenate the data to the main DataFrame
        data = pd.concat([data, df], axis=1)

    # Create a multi-index header
    data.columns = pd.MultiIndex.from_tuples(columns)
    return data

    # sample output from get_data function
###data = get_data(portfolio, start_date='2015-01-01', end_date='2023-01-01')
###data.interpolate(method = 'linear', inplace = True,limit_direction = 'both')
# print all the adjusted closing prices of the stocks in BRA
###data["BRA"]["Adj Close"]

def run_strategy_sys_eq(data, start_date, end_date):
    '''
    Given a trading list check to see if the portfolio weights sum to 0 and if the
    absolute value of the weights sum to 1.
    '''

    def check_long_short(trading_list):
        vals = trading_list.sum(axis = 1)
        abs_vals = trading_list.abs().sum(axis = 1)
        def is_close_to_zero(value, threshold=1e-6):
            return abs(value) <= threshold

        def is_close_to_one(value, threshold=(1+1e-6)):
            return abs(value) <= threshold

        # Apply the function to all elements in the Series
        result = vals.apply(lambda x: is_close_to_zero(x))
        result_ones = abs_vals.apply(lambda x: is_close_to_one(x))
        # Check if all values are close to 0
        all_close_to_zero = result.all()
        # Check if all values are close to 1
        all_close_to_one= result_ones.all()
        return all_close_to_zero, all_close_to_one


    '''
    check to see if the daily turn over is less than the treshold (0.25)
    '''
    def check_turn_over(trading_list, treshold=0.25):
        abs_turn_over = trading_list.diff().abs().sum(axis = 1)

        def is_close_to_turnover(value, threshold=(treshold+1e-6)):
            return abs(value) <= threshold

        # Apply the function to all elements in the Series
        result_turnover = abs_turn_over.apply(lambda x: is_close_to_turnover(x))
        # Check if all values are close to 0
        all_close_to_turnover= result_turnover.all()
        return all_close_to_turnover

    # Improved trading strategy with reduced turnover
    def reduced_turnover_strat(data, start_date, end_date, short_window=2, long_window=10, trade_threshold=0.01, decay_factor=0.9):
        trading_list = pd.DataFrame(index=data.index, columns=data.columns.levels[2])

        for country in data.columns.levels[0]:
            for ticker in data[country]['Adj Close'].columns:
                # Calculate the short and long moving averages
                short_ma = data[country]['Adj Close'][ticker].rolling(window=short_window).mean()
                long_ma = data[country]['Adj Close'][ticker].rolling(window=long_window).mean()

                # Create signals based on moving average crossovers
                signal = np.where(short_ma > long_ma, 1.0, 0.0)
                positions = np.where(signal == 1, 1, -1)  # 1 for long and -1 for short

                # Only trade if the difference between moving averages is more than the threshold
                ma_diff = short_ma - long_ma
                positions[np.abs(ma_diff) < trade_threshold] = 0

                # Apply decay factor to reduce turnover
                for i in range(1, len(positions)):
                    positions[i] = decay_factor * positions[i] + (1 - decay_factor) * positions[i - 1]

                trading_list[ticker] = positions

        # Normalize the portfolio weights
        trading_list_one = trading_list.sub(trading_list.mean(axis=1), axis=0)
        trading_list_normalized = trading_list_one.div(trading_list_one.abs().sum(axis=1), axis=0)

        # Adjust weights to reduce turnover
        for date in trading_list_normalized.index[1:]:
            prev_date = trading_list_normalized.index[trading_list_normalized.index.get_loc(date) - 1]
            diff = trading_list_normalized.loc[date] - trading_list_normalized.loc[prev_date]
            turnover = diff.abs().sum()

            if turnover > 0.25:
                alpha = 0.25 / turnover
                adjusted_weights = (1 - alpha) * trading_list_normalized.loc[prev_date] + alpha * trading_list_normalized.loc[date]
                trading_list_normalized.loc[date] = adjusted_weights
        return trading_list_normalized

    # Generate the new trading list using the refined strategy
    ###refined_trading_list = refined_strategy(portfolio, '2015-01-01', '2023-01-01')

    # Check if the refined strategy satisfies the turnover condition
    ###is_turnover_satisfied_refined = check_turn_over(refined_trading_list)
    ###is_turnover_satisfied_refined


    # Check the improved strategy with reduced turnover against the criteria
    #trading_list = reduced_turnover_strat(portfolio, start_date='2015-01-01',end_date='2023-01-01')
    trading_list = reduced_turnover_strat(data, start_date,end_date)
    #trading_list = modified_reduced_turnover_strat(portfolio, start_date='2015-01-01',end_date='2023-01-01')
    #trading_list = improved_reduced_turnover_strat(portfolio, start_date='2015-01-01',end_date='2023-01-01')

    #trading_list = refined_trading_list

    # this is what the output to our trading strategy looks like
    #trading_list = simple_strat(portfolio, start_date='2015-01-01',end_date='2023-01-01')
    #trading_list.head(5)
    check_turn_over(trading_list, 0.25)
    check_long_short(trading_list)
    ###trading_list= trading_list
    ###trading_list

    '''
    Overview: given a trading list of portfolio holdings use yfinance to find the
    overall pnl.

    # check PNL is not negative

    # if you loose all your money your done
    # make sure all dates are within bounds
    # make sure dates are every day
    # make sure dates are order
    # make sure weights sum to 0 and absolute value to 1
    #
    #weights after closing day
    '''


    def backtest(trading_list: pd.DataFrame,x_list, y_list) -> None:
        check_date(trading_list)
        check_weights(trading_list)
        check_pnl(trading_list,x_list, y_list)

    def check_weights(trading_list: pd.DataFrame) -> None:
        assert (trading_list.isna().all(axis=1) | trading_list.notna().all(axis=1)).all(), "There is a row with some, but not all NaN values"

        row_sums = abs(trading_list.sum(axis=1)) < 1e-5
        assert row_sums.all(), "Weights do not add up to zero"

        row_sums_abs = abs(trading_list.abs().sum(axis=1)) - 1 < 1e-5
        assert row_sums_abs.all(), "Absolute values of weights do not add up to one"

    def calculate_money(index, holdings, values):
        if holdings == {}:
            return sum(values.values())
        value = 0
        for stock in holdings:
            num_stock = holdings[stock]
            country = stock_mapping[stock]
            stock_value_eod = data[country]['Adj Close'].loc[index][stock]
            if math.isnan(stock_value_eod):
                value += values[stock]
            else:
                value += num_stock*stock_value_eod
        return value

    # Make Sure PNL is not negative and if at any moment your money is 0, you are done.
    def check_pnl(trading_list: pd.DataFrame,x_list, y_list) -> None:
        stock_holdings = defaultdict(float)
        stock_holdings_values = defaultdict(float)
        for stock in stock_mapping:
            stock_holdings_values[stock] = 100000/len(stock_mapping)

        for cur_day, row in trading_list.iterrows():
            if row.isna().all():
                continue

            cur_money = calculate_money(cur_day, stock_holdings,stock_holdings_values) # sold everything
            build = 0
            for stock, weight in row.items():
                num_stocks_value = cur_money * weight
                country = stock_mapping[stock]
                stock_value_eod = data[country]['Adj Close'].loc[cur_day][stock]
                if weight < 0:
                    stock_holdings[stock] = num_stocks_value / stock_value_eod if not math.isnan(stock_value_eod) else stock_holdings[stock]
                    stock_holdings_values[stock] = num_stocks_value
                else:
                    stock_holdings[stock] = 3*num_stocks_value / stock_value_eod if not math.isnan(stock_value_eod) else stock_holdings[stock]
                    stock_holdings_values[stock] = 3*num_stocks_value

                build += stock_holdings_values[stock]
            x_list.append(cur_day)
            y_list.append(build)
            assert build > 0 , "you lost all your money"

    def check_date(trading_list: pd.DataFrame) -> None:
        assert all(isinstance(idx, pd.Timestamp) for idx in trading_list.index), "An index is not a datetime object"

        start_date = pd.to_datetime('2015-01-01')
        end_date = pd.to_datetime('2023-01-01')
        all_within_timeframe = (trading_list.index >= start_date) & (trading_list.index <= end_date)
        assert all(all_within_timeframe), "You have elements outside of the date range"

        assert len(set(trading_list.index)) == len(trading_list.index), "Cannot have rows with duplicate dates"

        assert len(data.index) == len(trading_list.index), "Need to cover all days"

        assert list(trading_list.index) == list(data.index), "Rows must be in order"

    time_stamps = []
    pnl_on_day = []
    backtest(trading_list,time_stamps,pnl_on_day)


    plt.plot(time_stamps,pnl_on_day)
    plt.title('Timestamps vs. PNL')
    plt.xlabel('Timestamps')
    plt.ylabel('PNL')
    plt.show()

