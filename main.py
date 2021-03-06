import argparse
import logging
from markettrading.actor.deep_qlearning_actor import Deep_QLearning_Actor 
from markettrading.logger import config_level
from markettrading.environments.market_env import MarketEnvironment
from markettrading.environments.simplified_env import SimplifiedMarketEnv 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        action="store_const", dest="log_level", const=logging.DEBUG,
        default=logging.INFO,
    )
    parser.add_argument(
        '-v', '--verbose',
        action="store_const", dest="log_level", const=logging.INFO,
    )
    args = parser.parse_args()    
    config_level(level=args.log_level)

    # env = MarketEnvironment()
    # env = SimplifiedMarketEnv()
    env = SimplifiedMarketEnv(price_function='Sinoid')
    actor = Deep_QLearning_Actor(layers_and_act_func=[[10, 'relu'], [10, 'relu']],
                                 env=env) 
    actor.train()
    actor.test()

