import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import set_random_seed


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Train and evaluate the LSTM model.')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--no-show', action='store_true', help='Save the plot without opening a window.')
    parser.add_argument('--output-dir', type=Path, default=project_dir / 'outputs')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    set_random_seed(42)
    df = pd.read_csv(project_dir / 'TCS.NS.csv', parse_dates=['Date'])
    df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
    sequence_length = 60
    train_size = int((len(df) - sequence_length) * 0.8)
    test_start = sequence_length + train_size

    # Fit only on training history to keep held-out prices out of preprocessing.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[['Close']].iloc[:test_start])
    scaled_data = scaler.transform(df[['Close']]).astype(np.float32)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        Input(shape=(sequence_length, 1)),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1),
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f'Training sequences: {len(X_train)}; test sequences: {len(X_test)}')
    model.fit(X_train, y_train, epochs=args.epochs, batch_size=32, verbose=2)

    predicted = scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    print('MSE:', mean_squared_error(actual, predicted))
    print('RMSE:', mean_squared_error(actual, predicted) ** 0.5)
    print('R2 Score:', r2_score(actual, predicted))
    baseline = df['Close'].shift(1).iloc[test_start:].to_numpy()
    print('Previous-close baseline RMSE:', mean_squared_error(actual, baseline) ** 0.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dates = df['Date'].iloc[test_start:]
    pd.DataFrame({
        'Date': dates,
        'Actual': actual,
        'Predicted': predicted,
    }).to_csv(args.output_dir / 'lstm_predictions.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, actual, label='Actual Price')
    plt.plot(dates, predicted, label='Predicted Price')
    plt.title('TCS Stock Price Prediction (LSTM)')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (INR)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plot_path = args.output_dir / 'lstm_prediction.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Plot saved to: {plot_path}')
    if not args.no_show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    main()
