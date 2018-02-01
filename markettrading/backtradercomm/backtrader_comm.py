import logging
import os
import sys  
import threading

import numpy as np
import pandas as pd

#https://github.com/backtrader/backtrader
import backtrader as bt

#https://github.com/openai/gym
import gym
from gym import spaces

from enum import Enum, unique

@unique
class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
   
    @classmethod
    def size(cls):
        return 3

@unique
class Position(Enum):
    SHORT = 0
    LONG = 1
    IDLE = 2

class DefaultStrategy(bt.Strategy, threading.Thread):
    def __init__(self, price_set_event, new_action_event, action_callback, price_callback, new_state_callback, stake):
        super().__init__(args=(price_set_event, new_action_event, ))

        self._price_set_event = price_set_event 
        self._new_action_event = new_action_event
        self._action_callback = action_callback
        self._price_callback = price_callback
        self._new_state_callback = new_state_callback

        self._data_close = self.datas[0].close
        self._data_open = self.datas[0].open
        self._data_high = self.datas[0].high
        self._data_low = self.datas[0].low

#        # Add a MovingAverageSimple indicator
        self._slow_sma = bt.indicators.ExponentialMovingAverage(
            self._data_close, period=81)

        self._med_sma = bt.indicators.ExponentialMovingAverage(
            self._data_close, period=27)

        self._fast_sma = bt.indicators.ExponentialMovingAverage(
            self._data_close, period=9)

        self._order = None
        self._stake = stake

    def _state(self):
        return [self._data_open[0], self._data_high[0], self._data_low[0], self._data_close[0], self._slow_sma[0], self._med_sma[0], self._fast_sma[0]]

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return # Buy/Sell order submitted/accepted to/by broker - Nothing to do

        if order.status in [order.Completed]:
            if order.isbuy():
                 logging.debug("BUY EXECUTED at Price: {}, Cost: {}, Commision: {} " .format(order.executed.price, order.executed.value, order.executed.comm))
            else:  # Sell
                 logging.debug("SELL EXECUTED at Price: {}, Cost: {}, Commision: {} " .format(order.executed.price, order.executed.value, order.executed.comm))

            #notify backtrader that the price was set 
            self._price_callback(order.executed.price)
            self._price_set_event.set()
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.debug('Order Canceled/Margin/Rejected')

        self._order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        logging.debug('OPERATION PROFIT, GROSS %.2f, NET %.2f, VOL %.2f' .format(trade.pnl, trade.pnlcomm, self._stake))

    def next(self):
        logging.debug("New Candle Received. O: {}, H: {}, L: {}, C: {}" .format(self._data_open[0], self.data_high[0], self._data_low[0], self.data_close[0]))
        self._new_state_callback(self._state(), False) #state, done

        self._new_action_event.wait()
        action, cur_position = self._action_callback()
        self._new_action_event.clear()

        if self._order:
            return

        if action == Action.BUY and cur_position != Position.LONG:
            logging.debug("BUY CREATE")
            self._order = self.buy()
        elif action == Action.SELL and cur_position != Position.SHORT:
            logging.debug("SELL CREATE")
            self._order = self.sell()
        else: #ignoring buy when bought and sell when sold
            self._price_set_event.set()

    def run(self):
        pass

    def stop(self):
        self._new_state_callback(self._state(), True) #state, done
        self._price_set_event.set()

class DataFeed:
    def __init__(self, feed_name, start_date, end_date):
        self.feed_name = feed_name
        self.start_date = start_date
        self.end_date = end_date
    
class BacktraderCommunication(threading.Thread):
    #gym attributes
    action_space = spaces.Discrete(Action.size())
    observation_space = 8#spaces.Tuple

    def __init__(self, **kwargs):
        """
        Keyword Args:
            
            strategy=DefaultStrategy
            data_feed=DefaultFeed                       Details about the data feed
            stake=10                                 
            init_cash=10000
            broker_commission=0.001                     0.1%
        """

        strategy = DefaultStrategy
        if 'strategy' in kwargs: strategy = kwargs['strategy']
        
        data_feed = DataFeed('orcl-1995-2014.txt', '1995/1/4', '1996/12/30')
        if 'data_feed' in kwargs: data_feed = kwargs['data_feed']

        self._stake = 10
        if 'stake' in kwargs: self._stake = kwargs['stake']

        self._init_cash = 10000
        if 'init_cash' in kwargs: self._init_cash = kwargs['init_cash']

        self._broker_commission = 0.001 
        if 'broker_commission' in kwargs: self._broker_commission = kwargs['broker_commission']
       
        self._price_set_event = threading.Event()
        self._new_action_event = threading.Event()
        self._new_state_event = threading.Event()

        super().__init__(args=(self._price_set_event, self._new_action_event, self._new_state_event))

        self._cerebro = bt.Cerebro()
        self._cerebro.addstrategy(strategy, 
                                  price_set_event=self._price_set_event, 
                                  new_action_event=self._new_action_event, 
                                  action_callback=self.get_action_callback, 
                                  price_callback=self.set_price_callback, 
                                  new_state_callback=self.new_state_callback,
                                  stake=self._stake)
        
        abs_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        feed_path = os.path.join(abs_path, 'data/' + data_feed.feed_name)

        data = bt.feeds.YahooFinanceCSVData(
                dataname=feed_path,
                fromdate=pd.to_datetime(data_feed.start_date),
                todate=pd.to_datetime(data_feed.end_date),
#                timeframe=TimeFrame.Day,
                reverse=False)
        self._cerebro.adddata(data)
        
        self._cerebro.addsizer(bt.sizers.FixedSize, stake=self._stake)
        self._cerebro.broker.setcommission(commission=self._broker_commission)
    
    #callback function for strategy
    def get_action_callback(self):
        return self._action, self._position

    def set_price_callback(self, price):
        self._price = price 

    def new_state_callback(self, state, done):
        self._cur_obs_state = state
        self._done = done
        self._cur_obs_state.append(self._position.value)
        self._new_state_event.set()

    def _compute_reward(self):
        if self._position == Position.IDLE:
            if self._action == Action.BUY:    self._position = Position.LONG
            elif self._action == Action.SELL: self._position = Position.SHORT
            self._position_entry_price = self._price
            return 0
        elif self._position == Position.LONG:
            if self._action == Action.SELL: 
                self._position = Position.IDLE
                return self._stake*(self._price - self._position_entry_price)
            return 0 #Hold
        elif self._position == Position.SHORT:
            if self._action == Action.BUY:
                self._position = Position.IDLE
                return self._stake*(self._position_entry_price - self._price)
            return 0 #Hold

    def reset(self):
        self._position = Position.IDLE
        self._action = Action.HOLD 
        self._position_entry_price = None
        self._price = None
        self._done = False
        self._cur_obs_state = None

        self._cerebro.broker.setcash(self._init_cash)

        logging.info('Starting Portfolio Value: {}' .format(self._cerebro.broker.getvalue()))

        self.start()
        self._new_state_event.wait()
        self._new_state_event.clear()
        
        return self._cur_obs_state

    def step(self, action):
        if not any(np.isnan(self._cur_obs_state)): #ignore action when there is any Nan in the state
            self._action = Action(action)
       
        self._new_action_event.set()
        #wait for strategy to act
        self._price_set_event.wait()
        self._price_set_event.clear()

        self._new_state_event.wait()
        self._new_state_event.clear()

        return self._cur_obs_state, self._compute_reward(), self._done, None

    #called by thread start() method
    def run(self):
        self._cerebro.run()
        logging.info('Final Portfolio Value: {}' .format(self._cerebro.broker.getvalue()))

    def plot(self):
        self._cerebro.plot(style='candle', bardown='pink')


