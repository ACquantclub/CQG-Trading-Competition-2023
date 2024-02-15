import pandas as pd
import datetime

# Root symbol of the underlying stock or ETF, padded with spaces to 6 characters
# Expiration date, 6 digits in the format yymmdd
# Option type, either P or C, for put or call
# Strike price, as the price x 1000, front padded with 0s to 8 digits


def check_format(df: pd.DataFrame):
    if df.isnull().values.any():
        raise AssertionError("NaN in dataframe")
    try:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    except:
        raise AssertionError("Not a datetime")
    for i in range(df.shape[0]):
        row = df.iloc[i]
        assert row["Action"] in ["B", "S"], "Action not B or S"
        assert isinstance(
            row["Datetime"], datetime.datetime
        ), "Datetime not in recognizable format"
        ticker, rest = row["Option Symbol"].split(" ")
        assert ticker in ["VIX", "SPX"], "Ticker not VIX or SPX"
        assert len(rest) == 15, "Price not padded to 8 digits"
        date = rest[:6]
        try:
            yymmdd = datetime.datetime.strptime(date, "%y%m%d")
        except:
            raise AssertionError("Datetime not in recognizable format")
        type = rest[6]
        assert type in ["P", "C"], "Type of Option is not P or C"


# implementation of a margin account in the backtester so that competitors are forced
# to set aside a certain amount of their capital when selling calls and puts
# (depending upon the strike price for selling a put and the underlying's value for selling a call).
# The way this would ideally work is that when they sell an option, we just deduct the specified
# margin amount from the cash they have in their portfolio and keep it in a margin account.
# When these short options positions go away (either at expiration day or if they liquidate
# their position by buying the option back), the margin amount for that option will get
# transferred from the margin account back to their portfolio.


def main():
    df_dict_correct = {
        "Datetime": ["2023-09-06 09:10:37.517015"],
        "Option Symbol": ["VIX 141122P00019500"],
        "Action": ["B"],
        "Order size": [10],
    }
    df_correct = pd.DataFrame(df_dict_correct)

    df_dict_incorrect_datetime = {
        "Datetime": ["2023-096"],
        "Option Symbol": ["VIX 141122P00019500"],
        "Action": ["B"],
        "Order size": [10],
    }
    df_incorrect_datetime = pd.DataFrame(df_dict_incorrect_datetime)

    df_dict_incorrect_option_symbol = df_dict_incorrect_datetime = {
        "Datetime": ["2023-09-06"],
        "Option Symbol": ["VPX 141122P00019500"],
        "Action": ["B"],
        "Order size": [10],
    }

    df_incorrect_option_symbol = pd.DataFrame(df_dict_incorrect_option_symbol)

    df_dict_incorrect_option_symbol2 = df_dict_correct = {
        "Datetime": ["2023-09-06 09:10:37.517015"],
        "Option Symbol": ["VIX 141122P0001950"],
        "Action": ["B"],
        "Order size": [10],
    }

    df_incorrect_option_symbol2 = pd.DataFrame(df_dict_incorrect_option_symbol2)

    df_dict_incorrect_option_symbol3 = {
        "Datetime": ["2023-09-06 09:10:37.517015"],
        "Option Symbol": ["VIX 141122500019500"],
        "Action": ["B"],
        "Order size": [10],
    }

    df_incorrect_option_symbol3 = pd.DataFrame(df_dict_incorrect_option_symbol3)

    df_dict_incorrect_option_symbol4 = {
        "Datetime": ["2023-09-06 09:10:37.517015"],
        "Option Symbol": ["VIX 1R1122P00019500"],
        "Action": ["B"],
        "Order size": [10],
    }

    df_incorrect_option_symbol4 = pd.DataFrame(df_dict_incorrect_option_symbol4)

    df_dict_incorrect_NaN = {
        "Datetime": ["2023-09-06 09:10:37.517015"],
        "Option Symbol": ["VIX 141122P00019500"],
        "Action": ["B"],
        "Order size": [None],
    }

    df_incorrect_NaN = pd.DataFrame(df_dict_incorrect_NaN)

    dfs = [
        df_correct,
        df_incorrect_datetime,
        df_incorrect_option_symbol,
        df_incorrect_option_symbol2,
        df_incorrect_option_symbol3,
        df_incorrect_option_symbol4,
        df_incorrect_NaN,
    ]
    for i, df in enumerate(dfs):
        try:
            check_format(df)
            print(f"df: {i}")
            print("no error")
        except AssertionError as e:
            print(f"df: {i}")
            print(f"error: {e}")


if __name__ == "__main__":
    main()
