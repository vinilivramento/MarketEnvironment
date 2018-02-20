# Using Binance exchange API: https://github.com/sammchardy/python-binance
from binance.client import Client
from datetime import datetime

class BinanceAPI:
    @classmethod
    def get_historical_data(cls, symbol, start_date, end_date, interval):
        client = Client("", "") #no need for api key when getting historical data
        klines = client.get_historical_klines(symbol, interval, start_date, end_date)

        #timestamp is in miliseconds and datatime.fromtimestamp requires seconds
        with open('../data/Binance_{}_{}_{}_{}.csv'.format(symbol, interval, start_date, end_date), 'w') as file:
            file.write('Date,Open,High,Low,Close,Volume\n')
            for line in klines:
                file.write("{},{},{},{},{},{}\n".format(datetime.fromtimestamp(line[0]/1e3), line[1], line[2], line[3], line[4], line[5]))

# if __name__ == "__main__":
    # BinanceAPI.get_historical_data("BTCUSDT", "08-01-2017", "01-31-2018", Client.KLINE_INTERVAL_1DAY)
