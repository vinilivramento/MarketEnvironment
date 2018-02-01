import logging

def config_level(level):
    logging.basicConfig(#filename="output.log", 
                    level=level, 
                    format="%(asctime)s:%(levelname)s: %(message)s")

