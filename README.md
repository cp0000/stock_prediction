# Stock Prediction

## Overview

利用 LSTM 和 Transformer 来对股票的价格进行预测

[利用深度学习进行股票价格预测](https://cp0000.github.io/2022/02/16/stock-prediction/)

## Requirements

- Python 3.8+
- numpy
- matplotlib
- seaborn
- torch
- sklearn
- pandas
- plotly
- pandas_datareader


## Usage

1. Clone the repo
2. Pip install the requirements `pip install -r requirements.txt`
3. 获取指定股票的历史数据 `python fetch_stock_data.py QQQ 1/1/2010`
4. Train the model `python main.py QQQ transformer`
