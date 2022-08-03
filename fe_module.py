import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data_file_path = './data/omxs30c1_td_raw.csv'
features_path = './features/'
end_of_auction_timestamp = '2022-05-02 07:00:00.000000+00:00'
flash_crash_event_starting_timestamp = "2022-05-02 07:57:07.298000+00:00"
flash_crash_event_ending_timestamp = "2022-05-02 08:05:32.046000+00:00"
features_file_path = './features/'


def tick_gradient(start_tick, end_tick, start_prx, end_prx):
    return (end_prx - start_prx) / (end_tick - start_tick)


def read_data():
    df = pd.read_csv(data_file_path)
    df = df.sort_values(by='Timestamp')
    df = df[df['Timestamp'] >= end_of_auction_timestamp]
    df['SEQNUM'] = range(1, df.shape[0] + 1)

    return df


def generate_reward_feature(tick_data_frame):
    data_before_fc = tick_data_frame.loc[pd.to_datetime(tick_data_frame.index) < pd.to_datetime(
        flash_crash_event_starting_timestamp)]
    data_before_fc['VOL'] = data_before_fc['TRDPRC_1'].expanding(
        1).std().fillna(0)

    data_after_fc = tick_data_frame.loc[pd.to_datetime(tick_data_frame.index) >= pd.to_datetime(
        flash_crash_event_starting_timestamp)]
    data_after_fc['VOL'] = data_after_fc['TRDPRC_1'].expanding(
        1).std().fillna(0)

    reward_variable = pd.concat([data_before_fc['VOL'], data_after_fc['VOL']])

    return reward_variable


def feature_engineering_phase():
    df = read_data()
    trade_tick_data = df[df['EVENT_TYPE'] == 'trade']
    trade_tick_data = trade_tick_data[['Timestamp',
                                       'SEQNUM',
                                       'TRDPRC_1',
                                       'TRDVOL_1']]
    trade_tick_data['TNO'] = range(1, trade_tick_data.shape[0] + 1)
    trade_tick_data['TPT'] = trade_tick_data['TNO'] / trade_tick_data['SEQNUM']
    trade_tick_data['RETURNS'] = trade_tick_data['TRDPRC_1'].pct_change()
    trade_tick_data['C_RETURNS'] = (
        1 + trade_tick_data['RETURNS']).cumprod() - 1
    trade_tick_data['NT_RETURNS'] = trade_tick_data['RETURNS'].shift(-1)
    trade_tick_data['VOL'] = trade_tick_data['TRDPRC_1'].expanding(
        2).std()

    price_gradient = [0]
    cumulative_ud_ticks = [0]
    sequential_ud_ticks = [0]
    for i in range(1, trade_tick_data.shape[0]):
        price_gradient.append(np.abs(tick_gradient(trade_tick_data.iloc[i - 1, 1],
                                                   trade_tick_data.iloc[i, 1],
                                                   trade_tick_data.iloc[i - 1, 2],
                                                   trade_tick_data.iloc[i, 2])))
        if trade_tick_data.iloc[i, 6] > 0:
            cumulative_ud_ticks.append(cumulative_ud_ticks[-1] + 1)
            if sequential_ud_ticks[-1] > 0:
                sequential_ud_ticks.append(sequential_ud_ticks[-1] + 1)
            else:
                sequential_ud_ticks.append(1)
        elif trade_tick_data.iloc[i, 6] == 0:
            cumulative_ud_ticks.append(cumulative_ud_ticks[-1])
            sequential_ud_ticks.append(0)
        else:
            cumulative_ud_ticks.append(cumulative_ud_ticks[-1] - 1)
            if sequential_ud_ticks[-1] < 0:
                sequential_ud_ticks.append(sequential_ud_ticks[-1] - 1)
            else:
                sequential_ud_ticks.append(-1)

    trade_tick_data['CUDTICKS'] = cumulative_ud_ticks
    trade_tick_data['SUDTICKS'] = sequential_ud_ticks
    trade_tick_data['PRICE_GRADIENT'] = price_gradient
    trade_tick_data = trade_tick_data.set_index('Timestamp')

    trade_tick_data = trade_tick_data.dropna()

    trade_tick_data.to_csv(features_path + 'environment.csv')

    reward_variable = generate_reward_feature(trade_tick_data)
    trade_tick_data.insert(loc=len(trade_tick_data.columns),
                           column='REWARD', value=reward_variable)

    trade_tick_data.insert(loc=len(trade_tick_data.columns), column='LABEL', value=np.where(
        (pd.to_datetime(trade_tick_data.index) <= pd.to_datetime(flash_crash_event_ending_timestamp)) &
        (pd.to_datetime(trade_tick_data.index) >= pd.to_datetime(
            flash_crash_event_starting_timestamp)), 1, 0))

    env = trade_tick_data.drop(['SEQNUM',
                                'TNO',
                                'RETURNS',
                                'C_RETURNS',
                                'NT_RETURNS'], axis=1)
    env.to_csv(features_path + 'df_environment_vol.csv')

    x = env.drop(['REWARD', 'LABEL'], axis=1).to_numpy()
    y = env['LABEL']
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    with open(features_path + 'df_environment.pickle', 'wb') as f:
        pickle.dump(x, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(features_path + 'environment.pickle', 'wb') as f:
        pickle.dump(x_scaled, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(features_path + 'rewards.pickle', 'wb') as f:
        pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)
