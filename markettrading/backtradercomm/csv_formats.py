import backtrader as bt

#Formatting class for Binance CSV Data reading by Backtrader. See https://www.backtrader.com/docu/datafeed.html  
class BinanceCSVData(bt.feeds.GenericCSVData):
  params = (
        ('dtformat', ('%Y-%m-%d %H:%M:%S')),
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
        ('reverse', False)
    )

