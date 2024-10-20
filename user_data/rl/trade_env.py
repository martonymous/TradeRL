import os
import zipfile
import random

import pandas as pd
import numpy as np
from scipy.stats import entropy, differential_entropy
import talib as ta
import gymnasium as gym
from gymnasium import spaces

import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
import stable_baselines3.common.noise as ns


class TradingEnv(gym.Env):
    def __init__(self, data_sources, initial_balance=1000, max_stakes=1, window_size=288, base_currencies={'ETH': .0}, quote_currency='USDC', buy_fee=0.0005, sell_fee=0.0005, render_mode=None, mode='train'):
        super(TradingEnv, self).__init__()
        self.render_mode = render_mode
        self.mode = mode
        if self.mode == 'train':
            self.current_step = 0
        else:
            self.current_step = -1                         # Load data from the selected source

        # Environment parameters
        self.base_currencies = base_currencies
        self.quote_currency = quote_currency
        self.stakes = {i: None for i in range(0, max_stakes)} # Initialize empty stakes

        # Trading fees
        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        # list of indicators to use in observation space
        self.indicators = ['normalized_close', 'normalized_volume', 'entropy', 'std', 'atr', 'ema5', 'ema8', 'ema13', 'ema21', 'ma50', 'ma100', 'ma200']
        self.window_size = window_size

        # trade history storage
        self.trade_history = []
        
        if isinstance(data_sources, str):
            if os.path.isdir(data_sources):
                data_sources = [os.path.join(data_sources, f) for f in os.listdir(data_sources)]
            else:
                data_sources = [data_sources]

        if isinstance(data_sources, list):
            self.data_paths = data_sources
            self.data_sources = {source.split('\\')[-1].split('.')[0].split('_')[0]: pd.DataFrame() for i, source in enumerate(data_sources)}
        elif isinstance(data_sources, dict):
            self.data_sources = data_sources
            self.data_paths = list(data_sources.keys())
        else:
            raise ValueError("Data sources must be a list of file paths or a dictionary of dataframes.")
        self.source_selection_method = 'random'
        self.load_all_data_sources()

        self.initial_balance = np.float64(initial_balance)  # Starting balance
        self.balance = np.float64(initial_balance)          # Balance in quote currency
        self.max_stakes = max_stakes                        # Maximum number of stakes
        self.base_currency_balance = self.base_currencies[self.base_currency] * self.data['close'].values[-1]
        self.total_balance = self.balance + self.base_currency_balance
                
        # Observation space: [closing price, MAs, balance fraction,]
        num_obs_features = len(self.indicators) + 1 + self.max_stakes  # normalized price + normalized volume + 7 MAs + balance + per-stake current_roi
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_obs_features,), dtype=np.float32)
        
        # Action space: [sell/hold decisions for each stake (continuous, thresholded) + buy fraction]
        # All actions are continuous and interpreted within the environment logic.
        # First `max_stakes` entries are for sell/hold, last entry is buy fraction.
        self.action_space = spaces.Box(low=0, high=1, shape=(self.max_stakes + 2,), dtype=np.float32)

    def load_data_source(self):
        """Function to select and load a data source based on the selection method."""
        if self.source_selection_method == 'random':
            # Randomly select a data source from the list
            self.current_data_source = random.choice(self.data_paths)
        else:
            self.current_data_source = self.data_paths[0]  # For now, default selects the first source

        if self.current_data_source.split('\\')[-1].split('_')[0] in self.data_sources:
            self.data = self.data_sources[self.current_data_source.split('\\')[-1].split('.')[0].split('_')[0]].copy()
        else:
            # Load data (assume the data is a CSV path or a DataFrame)
            if isinstance(self.current_data_source, str):  # If the source is a file path
                if self.current_data_source.endswith('.csv'):
                    self.data = pd.read_csv(self.current_data_source)
                elif self.current_data_source.endswith('.feather'):
                    self.data = pd.read_feather(self.current_data_source)
            else:
                self.data = self.current_data_source.copy()  # If it's already a DataFrame

        # select base currency
        file_string = self.current_data_source.split('\\')[-1].split('.')[0].split('_')[0]
        if file_string in self.base_currencies:
            self.base_currency = file_string
        else:
            self.base_currency = list(self.base_currencies.keys())[0]

    def load_all_data_sources(self):
        data = None
        self.current_close_prices = {}
        for i, source in enumerate(self.data_paths):
            # Load the data source
            if isinstance(source, str):  # If it's a file path
                if source.endswith('.csv'):
                    data = pd.read_csv(source)
                elif source.endswith('.feather'):
                    data = pd.read_feather(source)

            elif isinstance(source, pd.DataFrame):  # If it's already a dataframe
                data = source.copy()

            else:
                raise ValueError("Data source must be a file path or a DataFrame.")
            
            self.data_sources[source.split('\\')[-1].split('.')[0].split('_')[0]] = data

            # add the base currency to the base_currencies dictionary
            if source.split('\\')[-1].split('_')[0] not in self.base_currencies:
                self.base_currencies[source.split('\\')[-1].split('.')[0].split('_')[0]] = .0

            # load the last timestep of the data into self.current_close_pricesa
            self.current_close_prices[source.split('\\')[-1].split('.')[0].split('_')[0]] = data['close'].values[-1]

        self.load_data_source()

    def process_data_sources(self, compute_indicators=True, zip_filename='processed_data.zip'):
        """Function to load all data sources, optionally compute indicators, and save to a zip file."""
        
        processed_dataframes = {}  # Store all processed dataframes
        # ensure that the data sources are not empty
        if len(self.data_sources) == 0:
            raise ValueError("Data sources are empty. Load data sources before processing.")
        
        for i, source in enumerate(self.data_sources):
            # Load the data source
            if isinstance(source, str):  # If it's a file path
                if source.endswith('.csv'):
                    data = pd.read_csv(source)
                elif source.endswith('.feather'):
                    data = pd.read_feather(source)
            elif isinstance(source, pd.DataFrame):  # If it's already a dataframe
                data = source.copy()  # If it's already a dataframe
            else:
                raise ValueError("Data source must be a file path or a DataFrame.")
            

            # Optionally compute indicators
            if compute_indicators:
                data = self.compute_indicators(data, self.window_size)

            # Store the processed dataframe
            split_filename = source.split('/')[-1]
            filename = split_filename.split('_')[0]
            processed_dataframes[f"{filename}.csv"] = data

        # Save all processed dataframes to a zip file
        self.save_to_zip(processed_dataframes, zip_filename)
        self.data_sources = [x for x in processed_dataframes.values()]  # Update data sources to processed data

    def save_to_zip(self, dataframes, zip_filename):
        """Save all processed dataframes into a zip file."""
        with zipfile.ZipFile(zip_filename, 'w') as zf:
            for filename, dataframe in dataframes.items():
                # Save each dataframe as a CSV file in the zip archive
                csv_data = dataframe.to_csv(index=False)
                zf.writestr(filename, csv_data)

    def load_from_zip(self, zip_filename):
        """Load dataframes from a zip file."""
        dataframes = {}
        with zipfile.ZipFile(zip_filename, 'r') as zf:
            for filename in zf.namelist():
                # Load each CSV file from the zip archive
                csv_data = zf.read(filename)
                dataframes[filename] = pd.read_csv(csv_data)
        return dataframes

    def compute_indicators(self, data, window_size=288):
        """Function to compute technical indicators and transformed variables for the environment.
    
            indicator data:
            - normalized close and volume
            - rolling entropy
            - rolling standard deviation (STD)
            - rolling average true range (ATR)
            - moving averages (MAs)
            - exponential moving averages (EMAs)
            
            TODO: Add more indicators as needed
        """

        # compute normalized close and volume
        data['normalized_close'] = self._rolling_normalize(data['close'], window_size=window_size)
        data['normalized_volume'] = self._rolling_normalize(data['volume'], window_size=window_size)

        # other rolling indicators
        data['entropy'] = self._compute_entropy(data['normalized_close'], window_size=window_size)
        data['atr'] = self._compute_atr(data, window_size=window_size)
        data['std'] = self._compute_std(data, window_size=window_size)

        # compute moving averages
        data['ma3'] = self._compute_moving_averages(data['normalized_close'], window_size=3)
        data['ma5'] = self._compute_moving_averages(data['normalized_close'], window_size=5)
        data['ma8'] = self._compute_moving_averages(data['normalized_close'], window_size=8)
        data['ma13'] = self._compute_moving_averages(data['normalized_close'], window_size=13)
        data['ma21'] = self._compute_moving_averages(data['normalized_close'], window_size=21)
        data['ma34'] = self._compute_moving_averages(data['normalized_close'], window_size=34)
        data['ma50'] = self._compute_moving_averages(data['normalized_close'], window_size=50)
        data['ma100'] = self._compute_moving_averages(data['normalized_close'], window_size=100)
        data['ma200'] = self._compute_moving_averages(data['normalized_close'], window_size=200)

        # compute exponential moving averages
        data['ema3'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=3)
        data['ema5'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=5)
        data['ema8'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=8)
        data['ema13'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=13)
        data['ema21'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=21)
        data['ema34'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=34)
        data['ema50'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=50)
        data['ema100'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=100)
        data['ema200'] = self._compute_exp_moving_averages(data['normalized_close'], window_size=200)

        # delete rows with NaN values
        data.dropna(inplace=True)
        return data

    def update_data(self, dataframe):
        """Function to update data for the environment in real-time."""
        self.data = dataframe.copy()

    def _compute_moving_averages(self, data, window_size=288):
        return ta.SMA(data, timeperiod=window_size)
    
    def _compute_exp_moving_averages(self, data, window_size=288):
        return ta.EMA(data, timeperiod=window_size)
    
    def _compute_rsi(self, data, window_size=288):
        return ta.RSI(data, timeperiod=window_size)
    
    def _compute_atr(self, data, window_size=288):
        return ta.ATR(data['high'], data['low'], data['close'], timeperiod=window_size)
    
    def _compute_entropy(self, data, window_size=288):
        return data.rolling(window=window_size, min_periods=1).apply(lambda x: differential_entropy(x))
    
    def _compute_std(self, data, window_size=288):
        return data['normalized_close'].rolling(window=self.window_size, min_periods=1).std()
    
    def _default_reward(self, **kwargs):
        reward = 1.0
        return float(reward)
    
    def _rolling_normalize(self, data, window_size=1000):
        rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
        rolling_std = data.rolling(window=window_size, min_periods=1).std()
        
        # Avoid division by zero
        rolling_std.replace(0, 1e-12, inplace=True)
        return (data - rolling_mean) / rolling_std
    
    def _get_indicator_data(self):
        return [self.data[x].values[self.current_step] for x in self.indicators]
         
    def _get_observation(self):
        current_price = self.data['normalized_close'].values[self.current_step]
        current_volume = self.data['normalized_volume'].values[self.current_step]
        indicators = self._get_indicator_data()
        
        balance_fraction = ((self.balance / self.total_balance) - 0.5) * 2 if self.total_balance > 0 else 0

        stakes_data = []
        for stake in self.stakes:
            if stake is not None:
                # Calculate stake ROI based on current price
                current_roi = (current_price - stake['entry_price']) / stake['entry_price'] - self.buy_fee
                stakes_data.append(np.float64(current_roi))
            else:
                stakes_data.append(0)  # For empty positions, return 0

        while len(stakes_data) < self.max_stakes:
            stakes_data.append(0)

        obs = np.array(indicators + [balance_fraction] + stakes_data, dtype=np.float64)  # Use float64 for observation

        # Clamp values if necessary to avoid overflow
        obs = np.clip(obs, -1e6, 1e6)  # Set reasonable bounds for your observation space

        # sanity checks
        assert not np.any(np.isinf(obs)), "Observation contains Inf values!"
        assert not np.any(np.isnan(obs)), "Observation contains NaN values!"
        
        return obs
    
    def _get_reward(self, **kwargs):
        # Assign the reward function, defaulting to self._default_reward if none provided
        reward_function = kwargs.get('reward_function', self._default_reward)

        if not callable(reward_function):
            raise ValueError("The provided reward_function or self._default_reward is not callable")

        reward = reward_function(self, **kwargs)

        if not isinstance(reward, (int, float)):
            raise ValueError("Reward must be a scalar value (int or float)")
        
        return reward
    
    def _set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.current_step = 0
        else:
            self.current_step = -1

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def step(self, action):
        """
            Step function for the environment, only in training mode.
            Action is a list of continuous values interpreted as follows:
            - First `max_stakes` entries are for sell/hold decisions for each stake (continuous, thresholded)
            - Last entry is the buy fraction (0: no buy, 1: buy)

        """

        sell_actions = action[:self.max_stakes]  # Actions for each stake (sell/hold)
        buy_decision = action[self.max_stakes]  # Buy decision (0: no buy, 1: buy)
        buy_fraction = 0
        
        if buy_decision >= .5:
            buy_fraction = action[self.max_stakes + 1]  # Fraction of balance to buy

        current_price = self.data['close'].values[self.current_step]
        profits = 0
        rois = 0

        # Handle selling stakes
        for i in range(len(self.stakes)):
            if sell_actions[i] <= 0 and self.stakes[i] is not None:  # Sell this stake if action signals and stake exists
                stake = self.stakes[i]  # Use positional indexing
                
                if stake is not None:  # Ensure valid stake
                    sale_price = stake['size'] * current_price
                    fee = sale_price * self.sell_fee
                    
                    # Calculate profit and reward
                    profit = sale_price - (stake['entry_price'] * stake['size']) - fee
                    roi = profit / (stake['entry_price'] * stake['size']) if stake['entry_price'] > 0 else 0
                    self.balance += sale_price - fee  # Add sale proceeds to balance

                    # Log trade history
                    self.trade_history.append({'quote_currency': self.quote_currency, 'step': self.current_step, 'type': 'sell', 'price': current_price, 'size': stake['size'], 'profit': profit, 'roi': roi, 'balance': self.balance})

                    # Clear the stake, keeping the placeholder
                    self.stakes[i] = None
                    profits += profit
                    rois += roi

        # Handle buying new stake, minimum buy fraction and minimum balance conditions
        min_buy_fraction = 0.02  # Minimum fraction of balance to buy
        min_stake_size = 0.01  # Minimum stake size
        min_balance = min_stake_size * current_price / (1 - self.buy_fee)  # Minimum balance to buy minimum stake size
        num_empty_positions = sum([1 for stake in self.stakes if stake is None])

        if buy_decision >= 0.5:
            if buy_fraction > min_buy_fraction and self.balance > min_balance:
                # Look for the first empty stake position (None)
                empty_position = None
                for idx, stake in enumerate(self.stakes):
                    if stake is None:
                        empty_position = idx
                        break

                if empty_position is not None:  # Only buy if there's an empty position
                    stake_to_buy = self.balance * buy_fraction
                    fee = stake_to_buy * self.buy_fee
                    size = (stake_to_buy - fee) / current_price
                    new_stake = {
                        'base_currency': self.base_currency,
                        'entry_amount': stake_to_buy - fee,
                        'entry_price': current_price,
                        'size': size
                    }
                    self.stakes[empty_position] = new_stake
                    self.balance -= stake_to_buy + fee  # Deduct buy amount and fee from balance
                    self.base_currencies[self.base_currency] += size  # Add to base currency balance
                    self.trade_history.append({'quote_currency': self.quote_currency, 'step': self.current_step, 'type': 'buy', 'price': current_price, 'size': size, 'balance': self.balance})

        # Update total balance
        self.base_currency_balance = sum([stake['size'] for stake in self.stakes])
        self.total_balance = self.balance + (self.base_currency_balance * current_price)

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        truncated = False

        # Get reward, next observation, and info dictionary
        rew_args = {
            'current_price': current_price,
            'buy_decision': buy_decision,
            'buy_fraction': buy_fraction,
            'num_empty_positions': num_empty_positions,
            'sell_actions': sell_actions,
            'stakes': self.stakes,
            'profits': profits,
            'rois': rois,
            'balance': self.balance,
            'done': done
        }
        reward = self._get_reward(**rew_args)
        obs = self._get_observation()
        info = self.trade_history[-1] if len(self.trade_history) > 0 else {}

        return obs, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        # Seed the environment (for reproducibility)
        super().reset(seed=seed)
        if options is not None:
            if options['step'] is not None:
                self.current_step = options['step']
            else:   
                self.current_step = np.random.randint(0, len(self.data) - 1)
        
        else:
            self.current_step = np.random.randint(0, len(self.data) - 1)

        self.balance = self.initial_balance
        self.base_currency_balance = 0  # Reset base currency balance to 0
        self.total_balance = self.initial_balance  # Reset total balance to initial balance
        self.stakes = []  # Reset stakes
        self.trade_history = []  # Reset trade history
        
        # Return observation and empty info dictionary
        return self._get_observation(), {}
    
    def render(self, mode='human'):
        if self.render_mode is not None:
            
            if self.render_mode == 'human':
                print(f"\tStep {self.current_step} --==-- Current Price: {self.data['close'].values[self.current_step]} === Balance: {self.balance} === Total Balance: {self.total_balance}")
                _ = [print(f"\t\tStake {i} -- Size: {stake['size']} -- Entry Price: {stake['entry_price']}") for i, stake in enumerate(self.stakes)]
                
            if self.render_mode == 'graph':
                # TODO: Implement graph rendering
                pass
            if self.render_mode == 'file':
                # TODO: Implement file logging for rendering
                pass
        else:
            pass



# Load Ethereum historical data from CSV
def load_data(file_path):
    data = pd.read_feather(file_path)
    data.set_index('date', inplace=True)
    return data

if __name__ == '__main__':

    """
    
    WINDOW SIZE TABLE:

    1m
    1h = 60
    4h = 240
    12h = 720
    1d = 1440

    3m 
    1h = 20
    4h = 80
    12h = 240
    1d = 480

    5m
    1h = 12
    4h = 48
    12h = 144
    1d = 288
    3d = 864

    15m
    6h = 24
    12h = 48
    1d = 96
    3d = 288
    1w = 672

    30m
    12h = 24
    1d = 48
    3d = 144
    1w = 336
    2w = 672

    1h
    1d = 24
    3d = 72
    1w = 168
    2w = 336
    1m = 720
    
    """

    list_of_data_sources = [
        "./user_data/data/binance/ADA_USDT-30m.feather",
        "./user_data/data/binance/AVAX_USDT-30m.feather",
        "./user_data/data/binance/BTC_USDT-30m.feather",
        "./user_data/data/binance/DOGE_USDT-30m.feather",
        "./user_data/data/binance/DOT_USDT-30m.feather",
        "./user_data/data/binance/ETH_USDT-30m.feather",
        "./user_data/data/binance/FET_USDT-30m.feather",
        "./user_data/data/binance/XRP_USDT-30m.feather",
        ]
    data_dir = "./user_data/ml/preprocessed/30m_processed"
    
    env = TradingEnv(data_dir, window_size=144, render_mode='human', mode='train')
    # env.process_data_sources(zip_filename='30m_processed.zip')	
    print(env.data)
    print(env.data_sources)

    train = False
    if train:
        pass
        # Wrap the environment using a DummyVecEnv for compatibility with Stable-Baselines
        # env = DummyVecEnv([lambda: TradingEnv(list_of_data_sources, render_mode='human', mode='train')])

        # Create the PPO model
        # model = PPO("MlpPolicy", env, gamma=0.9999, device='cuda', learning_rate=0.0001, n_steps=4032, batch_size=32, n_epochs=1, tensorboard_log="./ppo_trading_tensorboard/")
        # model = SAC("MlpPolicy", 
        #             env, 
        #             gamma=0.9999, 
        #             device='cpu', 
        #             learning_rate=0.0001, 
        #             buffer_size=1000, 
        #             batch_size=1024,
        #             tau=0.005,
        #             learning_starts=200,
        #             train_freq=3,
        #             target_update_interval=10,
        #             action_noise=ns.NormalActionNoise(mean=0.5 * np.ones(1), sigma=0.5 * np.ones(1)),
        #             verbose=1, 
        #             tensorboard_log="./sac_trading_tensorboard/")
        
        # # Verify the model is using the GPU
        # print(model.policy.device)

        # Set up custom logging with SB3's logger
        # new_logger = configure("./ppo_trading_tensorboard/", ["stdout", "tensorboard"])
        # newsac_logger = configure("./ppo_trading_tensorboard/", ["stdout", "tensorboard"])

        # Apply logger to the PPO model
        # model.set_logger(new_logger)

        # Train the model
        # model.learn(total_timesteps=12000, log_interval=1, progress_bar=True)

        # # Save the trained model
        # model.save("sac_eth_trading_model")
        # # Reset environment, set a random initial step

        # initial_step = np.random.randint(0, len(eth_data) - 1)
        # env.envs[0].current_step = 0  # Directly set the current step
        # env.envs[0].current_step = initial_step  # Directly set the current step

        # Now reset the environment
        # obs = env.reset()
        # for i in range(1000):
        #     action, _states = model.predict(obs)
        #     obs, rewards, done, info = env.step(action)
        #     if i % 10 == 0:
        #         env.render()
        #         print(f"Step: {i} -- Reward: {rewards} -- Done: {done}")
        #     if done:
        #         break

        # Close the environment
        # env.close()
        pass

