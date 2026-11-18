import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Train and evaluate the linear regression model.')
    parser.add_argument('--no-show', action='store_true', help='Save the plot without opening a window.')
    parser.add_argument('--output-dir', type=Path, default=project_dir / 'outputs')
    args = parser.parse_args()
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = pd.read_csv(project_dir / 'TCS.NS.csv', parse_dates=['Date'])
    df = df.dropna(subset=['Date', 'Close']).sort_values('Date').reset_index(drop=True)
    df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)
    X = df[['Date_ordinal']]
    y = df['Close']

    train_size = int(len(df) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f'Training rows: {len(X_train)}; test rows: {len(X_test)}')
    print('MSE:', mean_squared_error(y_test, y_pred))
    print('RMSE:', mean_squared_error(y_test, y_pred) ** 0.5)
    print('R2 Score:', r2_score(y_test, y_pred))
    print('Previous-close baseline RMSE:', mean_squared_error(y_test, y.shift(1)[train_size:]) ** 0.5)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'Date': df['Date'][train_size:],
        'Actual': y_test,
        'Predicted': y_pred,
    }).to_csv(args.output_dir / 'linear_predictions.csv', index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Actual')
    plt.plot(df['Date'][train_size:], y_pred, label='Predicted', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.title('TCS Stock Price Prediction (Linear Regression)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plot_path = args.output_dir / 'linear_prediction.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Plot saved to: {plot_path}')
    if not args.no_show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    main()
