import pandas as pd
import httpx

import settings


class DataGenerator:
    def __init__(self, symbol: str):
        # Add logic here to check if we already have the latest data
        self.symbol = symbol

    def _get_techninal_indicator_daily_time_series(
        self,
        indicator: str,
    ) -> pd.DataFrame:
        params = {
            'apikey': settings.apikey,
            'symbol': self.symbol,
            'function': indicator,
            'interval': settings.interval,
        }

        if indicator == 'AD':
            indicator = 'Chaikin A/D'

        if indicator in ['SMA', 'WMA', 'RSI', 'BBANDS', 'TRIX']:
            params['time_period'] = settings.time_period
            params['series_type'] = settings.series_type

        if indicator in ['DX', 'MFI', 'AROON', 'ADX']:
            params['time_period'] = settings.time_period

        if indicator in ['MACD', 'PPO']:
            params['series_type'] = settings.series_type

        json_response = httpx.get(
            url='https://www.alphavantage.co/query',
            params=params
        ).json()
        time_series = []

        for date, data in json_response[f"Technical Analysis: {indicator}"].items():
            if indicator == 'STOCH':
                time_series.append(
                {
                    "date": date,
                    "SlowK": float(data["SlowK"]),
                    "SlowD": float(data["SlowD"])
                }
            )
            elif indicator == 'AROON':
                time_series.append(
                {
                    "date": date,
                    "AroonDown": float(data["Aroon Down"]),
                    "AroonUp": float(data["Aroon Up"])
                }
            )
            elif indicator == 'MACD':
                time_series.append(
                {
                    "date": date,
                    "MACD": float(data["MACD"]),
                    "MACD_Signal": float(data["MACD_Signal"]),
                    "MACD_Hist": float(data["MACD_Hist"]),
                }
            )
            elif indicator == 'BBANDS':
                time_series.append(
                {
                    "date": date,
                    "Real_Upper_Band": float(data["Real Upper Band"]),
                    "Real_Lower_Band": float(data["Real Lower Band"])
                }
                )
            else:
                time_series.append(
                    {
                        "date": date,
                        f"{indicator}": float(data[indicator])
                    }
                )

        return pd.DataFrame(time_series)


    def _get_crypto_daily_time_series(self, market: str = 'USD') -> pd.DataFrame:
        json_response = httpx.get(f'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={self.symbol}&market={market}&apikey=KNPL6J9N740SLRRG').json()
        time_series = []

        for date, data in json_response["Time Series (Digital Currency Daily)"].items():
            time_series.append(
                {
                    "date": date,
                    "open": float(data["1. open"]),
                    "high": float(data["2. high"]),
                    "low": float(data["3. low"]),
                    "close": float(data["4. close"]),
                    "volume": float(data["5. volume"]),
                }
            )

        return pd.DataFrame(time_series)


    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data['OBV_pct_change'] = data['OBV'].pct_change() * 100
        data['AD_pct_change'] = data['Chaikin A/D'].pct_change() * 100
        data['TRIX'] = data['TRIX'] * 100
        data['BBANDS_distance_pct'] = ((data['Real_Upper_Band'] - data['Real_Lower_Band']) / data['Real_Lower_Band']) * 100
        data.drop(columns=['OBV', 'Chaikin A/D', 'Real_Upper_Band', 'Real_Lower_Band', 'close', 'volume', 'open', 'high', 'low', 'date'], axis=1, inplace=True)
        data.dropna(inplace=True)
        return data.astype(float)


    def _fetch_data(self) -> pd.DataFrame:
        if self.symbol in ['BTC', 'ETH', 'SOL', 'ADA', 'TRX', 'MATIC', 'LTC', 'UNI', 'ATOM', 'MKR', 'GRT', 'SNX', 'NEO', 'GNO', 'ALGO']:
            market = 'EUR'
        else:
            market = 'USD'

        time_series_df = self._get_crypto_daily_time_series(market)
        self.symbol = f"{self.symbol}USDT"

        indicators_dfs = []
        for indicator in settings.indicators:
            indicators_dfs.append(self._get_techninal_indicator_daily_time_series(indicator))

        merged_df = pd.merge(time_series_df, indicators_dfs[0], on='date')
        for df in indicators_dfs[1:]:
            merged_df = pd.merge(merged_df, df, on='date')

        merged_df['date'] = pd.to_datetime(merged_df['date'])
        merged_df.sort_values(by='date', inplace=True)
        merged_df.reset_index(inplace=True, drop=True)
        return merged_df


    def get_dataset(
        self,
        look_ahead_days: int = settings.prediction_window_days,
        downtrend: bool = False
    ) -> pd.DataFrame:
        """
        Returns the dataset that will be user for training and evaluation of the models
        Params:
        - look_ahead_days: the prediction timeframe
        - downtrend: If True the target variable will contain 1 if the price will go down,
        If False the target variable will contain 1 if the price will go up
        """
        data = self._fetch_data()
        
        # Creata a new column with the target variable        
        if downtrend:
            data['min_in_next_window_days'] = data['low'].rolling(window=look_ahead_days).min().shift(-look_ahead_days + 1)
            data.dropna(inplace=True)
            percentage_difference = (data['min_in_next_window_days'] - data['close']) / data['close']
            data['target'] = (percentage_difference >= settings.target_downtrend_pct).astype(int)
        else:
            data['max_in_next_window_days'] = data['high'].rolling(window=look_ahead_days).max().shift(-look_ahead_days + 1)
            data.dropna(inplace=True)
            percentage_difference = (data['max_in_next_window_days'] - data['close']) / data['close']
            data['target'] = (percentage_difference >= settings.target_uptrend_pct).astype(int)
        
        data.drop(columns=['max_in_next_window_days',], axis=1, inplace=True)
        data = self._transform_data(data)
        data.dropna(inplace=True)
        
        return data


    def get_prediction_input(self) -> pd.DataFrame:
        data = self._fetch_data()
        data = self._transform_data(data)
        return data.iloc[-1:]
