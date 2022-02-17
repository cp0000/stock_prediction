# Standard packages
import pandas as pd
import argparse
from distutils.util import strtobool

# Scripts
from preprocess import (
    config,
    get_timestamps,
    collect_data,
    plot_closing,
    plot_gain,
    compare_stocks,
)
from models import TorchRNN, rnn_params, transf_params, TransformerModel
from dataset import GetDataset
from train import Classifier, plot_predictions
import Analysis


def visualization():
    for idx, stock in enumerate(config.stock_names):
        timestamps = get_timestamps(config.yrs, config.mths, config.dys)
        df = collect_data(timestamps, stock, config.moving_averages, True)
        fig1 = plot_closing(df, moving_averages=True, intervals=None)
        fig1.show()
        # fig2 = plot_gain(df)
        # fig2.show()
        daily_returns, fig1_c, fig2_c = compare_stocks(
            config.stock_names_compare, timestamps
        )


def run(stock: str, model_type: str, stationary=True):
    print(stock, model_type, stationary)

    stock = f"./Data/{stock}.csv"  # stock_csv_filepath

    df = Analysis.get_data(stock)
    df["Company stock name"] = stock.split("/")[-1].split(".")[0]
    dataset = GetDataset(df)
    dataset.get_dataset(scale=False, stationary=stationary)
    train_data, test_data, train_data_len = dataset.split(
        train_split_ratio=0.8, time_period=15
    )
    train_data, test_data = dataset.get_torchdata()
    x_train, y_train = train_data
    x_test, y_test = test_data

    if model_type == "lstm":
        params = rnn_params
        model = TorchRNN(
            rnn_type=params.rnn_type,
            input_dim=params.input_dim,
            hidden_dim=params.hidden_dim,
            output_dim=params.output_dim,
            num_layers=params.num_layers,
        )
    elif model_type == "transformer":
        params = transf_params
        model = TransformerModel(params)
    else:
        raise ValueError(
            'Wrong model type selection, select either "rnn" or "transformer"!'
        )

    clf = Classifier(model)
    clf.train([x_train, y_train], params=params)
    y_scaler = dataset.y_scaler
    print('y_scaler', y_scaler)

    print('x_test',len(x_test))
    print('y_test',len(y_test))

    predictions = clf.predict([x_test, y_test], y_scaler, data_scaled=False)
    predictions = pd.DataFrame(predictions)
    predictions.reset_index(drop=True, inplace=True)
    predictions.index = df.index[-len(x_test) :]
    # predictions["Actual"] = y_test[:-1]
    predictions["Actual"] = y_test[:]

    predictions.rename(columns={0: "Predictions"}, inplace=True)
    if stationary:
        predictions = Analysis.inverse_stationary_data(
            old_df=df,
            new_df=predictions,
            orig_feature="Actual",
            new_feature="Predictions",
            diff=12,
            do_orig=False,
        )
    stock_symbol = df["Company stock name"].values[0]
    plot_predictions(df, train_data_len, predictions["Predictions"].values, model_type)
    trade_stock(df, train_data_len, predictions["Predictions"].values, stock_symbol, model_type)

def trade_stock(df, train_data_len, predictions, stock, model_type):
    valid = df[train_data_len:][:-2]

    buys    = []
    sells   = []
    thresh  = 0.001
    x       = -1

    for index, ind in enumerate(valid['Close']):
        next_day_predictions = predictions[index]
        delta = (next_day_predictions - ind) / ind
        if delta > thresh:
            buys.append((x, ind))
        elif delta < -thresh:
            sells.append((x, ind))
        x += 1
    print(f"buys: {len(buys)}")
    print(f"sells: {len(sells)}")

    # we create new lists so we dont modify the original
    compute_earnings([b for b in buys], [s for s in sells])

    import matplotlib.pyplot as plt

    plt.gcf().set_size_inches(22, 15, forward=True)

    real = plt.plot(valid['Close'].values, label='real')
    pred = plt.plot(predictions, label='predicted')

    if len(buys) > 0:
        plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00', s=50)
    if len(sells) > 0:
        plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000', s=50)

    plt.legend(['Real', 'Predicted', 'Buy', 'Sell'])
    plt.savefig(
        f"./demonstration_images/{stock}_{model_type}_trade.svg"
    )

def compute_earnings(buys_, sells_):
    purchase_amt = 100
    stock = 0
    balance = 0
    while len(buys_) > 0 and len(sells_) > 0:
        if buys_[0][0] < sells_[0][0]:
            # time to buy $10 worth of stock
            balance -= purchase_amt
            stock += purchase_amt / buys_[0][1]
            buys_.pop(0)
        else:
            # time to sell all of our stock
            balance += stock * sells_[0][1]
            stock = 0
            sells_.pop(0)
    print(f"earnings: ${balance}")

if __name__ == "__main__":
    # visualization()
    # run("./Data/TSLA.csv", "transformer", True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stock",
        type=str,
        help="the stock symbol you want to download, e.g. MSFT,AAPL,GOOG",
    )
    parser.add_argument(
        "model_type",
        type=str,
        choices=["lstm", "transformer"],
        help="lstm or transformer",
    )
    parser.add_argument(
        "stationary",
        type=lambda x: bool(strtobool(x)),
        nargs="?",
        const=True,
        default=False,
    )

    args = parser.parse_args()
    print(args)

    stock_symbol = args.stock
    model_type = args.model_type
    stationary = args.stationary

    run(stock_symbol, model_type, stationary)
