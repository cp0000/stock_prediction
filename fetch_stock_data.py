import plotly.graph_objects as go
import pandas_datareader.data as web
from datetime import datetime
from pprint import pprint
import json
import argparse


def fetch_stock_dataset(symbol, start_time):
    print(symbol, start_time)
    df = web.DataReader(symbol, data_source="yahoo", start=start_time)
    pprint(df.head(10))

    """
    visualization fetch data
    graph = {
        'x': df.index,
        'open': df.Open,
        'close': df.Close,
        'high': df.High,
        'low': df.Low,
        'type': 'candlestick',
        'name': 'BILI',
        'showlegend': True
    }
    layout = go.Figure(
        data = [graph],
        layout_title="BILI Stock"
    )
    layout.show()
    """

    df.to_csv(f"./Data/{symbol}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "symbol",
        type=str,
        help="the stock symbol you want to download, e.g. MSFT,AAPL,GOOG",
    )
    parser.add_argument(
        "start_time",
        type=str,
        help="Timestamp left boundary for range (defaults to 1/1/2010)",
    )

    namespace = parser.parse_args()
    fetch_stock_dataset(**vars(namespace))
