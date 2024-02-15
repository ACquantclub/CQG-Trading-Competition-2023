import pandas as pd

import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import random
import pickle
from sklearn import svm
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

#importing data !
#b1 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures1.csv')
b2 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures2.csv')
#b3 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures3.csv')
#b4 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures4.csv')
#b5 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures5.csv')
#b6 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures6.csv')
#b7 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures7.csv')
#b8 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures8.csv')
#b9 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures9.csv')
#b10 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures10.csv')
#b11 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures11.csv')
#b12 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures12.csv')
#b13 = pd.read_csv('/Users/sebastienbrown/Documents/GitHub/TraderMammoths/Case1/CTC23_Blockchain_Data/BTC_Futures13.csv')
#b14 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures14.csv')
#b15 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures15.csv')
#b16 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures16.csv')
#b17 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures17.csv')
#b18 = pd.read_csv('/Users/sebastienbrown/Documents/GitHub/TraderMammoths/Case1/CTC23_Blockchain_Data/BTC_Futures18.csv')
#b19 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures19.csv')
#b20 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures20.csv')
#b21 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures21.csv')
#b22 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures22.csv')
#b23 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures23.csv')
#b24 = pd.read_csv('CTC23_Blockchain_Data/BTC_Futures24.csv')

def run_strategy_crypto(filelist):


    '''
    Given a dataframe of positions check that the dates and positions are valid.
    '''


    def check_crypto_output(marketdata, positions):
        # check if positions is a dataframe
        assert isinstance(positions, pd.DataFrame), "positions should be a dataframe"
        assert "DATETIME" in positions.columns, "positions dataframe does not have 'DATETIME' column, please read naming specifications"

        # check whether every value in 'DATETIME' is a datetime object
        assert positions['DATETIME'].apply(lambda x: isinstance(x, pd.Timestamp)).all(), "every element in 'DATETIME' column of positions should be a datetime object"

        # check if right number of dates, and that they are equal
        assert marketdata['ts_event'].equals(positions['DATETIME']), "the 'DATETIME' column of positions should match 'ts_recv' of marketdata column"

        # check if all outputs are valid
        assert all(positions['POSITION'].isin([-1, 0, 1])), "all values in 'DATETIME' column need to be either -1, 0 or 1"

    '''
    Overview: given a list of positions use provided market data to find the
    overall pnl.
    '''

    def backtest(marketdata: pd.DataFrame, positions: pd.DataFrame, y_list) -> None:
        check_crypto_output(marketdata, positions)
        return check_pnl(marketdata, positions, y_list)


    def check_pnl(marketdata: pd.DataFrame, positions: pd.DataFrame, y_list) -> None:
        pnl = 0  # inital capital is 0 dollars
        curpos = 0 # setting initial position to neutral
        spread_cost = 0 # track total spread

        for index, row in marketdata.iterrows():
            bid_price = row['bid_px_00'] / 1e-9
            ask_price = row['ask_px_00'] / 1e-9
            signal = positions.loc[index, 'POSITION'] # whether we buy or sell
            #print(signal)

            # calculate spread cost
            spread = (ask_price - bid_price)/2

            #Note: You effectively trade at the midpoint at each time period,
            #and are compensated for the spread when you both open and close a position.

            # return to neutral
            if curpos == -1:
                pnl -= ask_price
            elif curpos == 1:
                pnl += bid_price

            # add spread
            if curpos != 0:
                spread_cost += spread

            # perform trade
            if signal == 1:
                pnl -= ask_price
            elif signal == -1:
                pnl += bid_price

            # add spread
            if signal != 0:
                spread_cost += spread

            # update position
            curpos = signal


            #Calculate PNL if we were to close - for graph
            pnl_close=pnl
            spread_close=spread_cost

            if curpos == -1:
                pnl_close -= ask_price
            elif curpos == 1:
                pnl_close += bid_price
            if curpos != 0:
                spread_close += spread

            y_list.append(pnl_close+spread_close)


        # return to neutral
        if curpos == -1:
            pnl -= ask_price
        elif curpos == 1:
            pnl += bid_price

        # add spread
        if curpos != 0:
            spread_cost += spread

        return (pnl + spread_cost)

    #print(filelist)
    data = pd.concat(filelist, ignore_index=True)
    datacopy=data
    scaler=StandardScaler()

    #data['bid_px_00']=(pd.to_numeric(data['bid_px_00'],errors="coerce"))
    data['ask_px_00']=(pd.to_numeric(data['ask_px_00'],errors="coerce"))

    data['ask_px_00'].fillna(method='ffill', inplace=True)
    #data=data.reset_index()
    #data['bid_px_00']=data['bid_px_00']

    # Example: Calculate rolling average of bid prices over a window of 100 time points
    data['rolling_avg_bid'] = data['bid_px_00'].rolling(window=100).mean()
    data['rolling_avg_bid'].fillna(method='bfill', inplace=True)

    # Price difference between consecutive time points
    data['price_diff'] = data['bid_px_00'].diff()
    data['price_diff'].fillna(method='bfill', inplace=True)

    data['ask_sz_00'] = data['ask_sz_00'].replace(0.0, 0.1)
    #print(data['ask_sz_00'])

    # Ratio of bid size to ask size
    data['bid_ask_ratio'] = data['bid_sz_00'] / data['ask_sz_00']

    #calculate the rolling max bid and ask sizes
    data['best_bid_size']=data['bid_sz_00'].rolling(window=100).max()
    data['best_ask_size']=data['ask_sz_00'].rolling(window=100).max()
    data['best_ask_size'].fillna(method='bfill', inplace=True)
    data['best_bid_size'].fillna(method='bfill', inplace=True)

    data['bid_size_metric']=(data['best_bid_size']-data['best_ask_size'])/(data['best_bid_size']+data['best_ask_size'])

    data['volume']=data['ask_sz_00']+data['bid_sz_00']

    short_window = 20
    long_window = 100

    data['short_mavg'] = data['bid_px_00'].rolling(window=short_window).mean()
    data['long_mavg'] = data['bid_px_00'].rolling(window=long_window).mean()
    data['short_mavg'].fillna(method='bfill', inplace=True)
    data['long_mavg'].fillna(method='bfill', inplace=True)


    features=scaler.fit_transform(data[['rolling_avg_bid',"bid_ask_ratio","bid_size_metric","volume","short_mavg","long_mavg"]])
    features=pd.DataFrame(features)

    data['rolling_avg_bid']=features[0]
    data['bid_ask_ratio']=features[1]
    data['bid_size_metric']=features[2]
    data['volume']=features[3]
    data['short_mavg']=features[4]
    data['long_mavg']=features[5]
    #print(data)

    #features=pd.DataFrame(features)
    # Splitting data into training and test set with normalized features
    data = data[['ts_event','bid_px_00','ask_px_00','rolling_avg_bid', 'price_diff', 'bid_ask_ratio','volume','short_mavg','long_mavg']]


    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)

    # Making predictions on the test set
    predictions = clf.predict(data)

    data['ts_event'] = data['ts_event'].apply(lambda x: datetime.utcfromtimestamp(x / 1000000000.0))
    

    print("data print event",data['ts_event'])
    positions = pd.DataFrame({
        "DATETIME": data['ts_event'],
        "POSITION": predictions #[random.choice([-1, 0, 1]) for _ in range(len(X_test))]
    })
    #print(positions['DATETIME'])
    #positions=positions.reset_index()
    return positions
