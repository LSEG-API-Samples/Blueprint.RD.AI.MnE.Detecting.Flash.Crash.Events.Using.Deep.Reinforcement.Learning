import pandas as pd
import enum

from tensorforce import Environment
from tensorforce.agents import Agent
from fe_module import features_file_path, flash_crash_event_starting_timestamp, flash_crash_event_ending_timestamp


class Decision(enum.Enum):
    NORMAL = 0
    ABNORMAL = 1


class ExchangeEnvironment(Environment):
    def __init__(self, environment_observations):
        super().__init__()
        self.state_iterator = 0
        self.reward_var = environment_observations['REWARD']
        self.environment_observations = environment_observations.iloc[:, :-2]
        self.no_states = self.environment_observations.shape[0]
        self.current_state = {}

    def states(self):
        state_dict = dict(type='float', shape=(7,))
        return state_dict

    def actions(self):
        return {"DECISION": dict(type="int", num_values=2)}

    def close(self):
        super().close()

    def reset(self):
        self.state_iterator = 0
        self.current_state = self.environment_observations.iloc[self.state_iterator]
        return self.current_state

    def max_episode_timesteps(self):
        return self.no_states

    def check_terminal_criteria(self, actions, all_decisions, timestamp, early_terminate=False):
        terminate = False
        cutoff_timestamp = '2022-05-02 07:58:20.119000+00:00'

        if early_terminate:
            if pd.to_datetime(timestamp) < pd.to_datetime(flash_crash_event_starting_timestamp) and \
                    actions['DECISION'] == 1:
                terminate = True
            elif pd.to_datetime(timestamp) > pd.to_datetime(cutoff_timestamp) and \
                    sum(all_decisions) == 0:
                terminate = True
        if self.state_iterator == self.no_states - 1:
            terminate = True
        return terminate

    def calculate_reward(self, reward_val, timestamp, decision, terminal):
        if terminal is True and pd.to_datetime(timestamp) < pd.to_datetime(flash_crash_event_ending_timestamp):
            reward = -200
        else:
            if pd.to_datetime(timestamp) < pd.to_datetime(flash_crash_event_starting_timestamp):
                if decision == Decision.NORMAL.value:
                    reward = reward_val
                else:
                    reward = 0

            elif pd.to_datetime(flash_crash_event_starting_timestamp) <= pd.to_datetime(timestamp) <= \
                    pd.to_datetime(flash_crash_event_ending_timestamp):
                if decision == Decision.NORMAL.value:
                    reward = -reward_val
                else:
                    reward = 0

            elif pd.to_datetime(timestamp) > pd.to_datetime(flash_crash_event_ending_timestamp):
                if decision == Decision.ABNORMAL.value:
                    reward = -3*reward_val
                else:
                    reward = 0

        return reward

    def report_accuracy(self, all_decisions, y):
        y_before = y[:y.index(1)]
        y_during = y[y.index(1): y[y.index(1):].index(0)+len(y_before)]
        y_after = y[::-1][:y[::-1].index(1)]

        accuracy = {}
        true_before = 0
        true_during = 0
        true_after = 0

        for i in range(len(y)):
            if all_decisions[i] == y[i]:
                if i <= len(y_before):
                    true_before += 1
                elif len(y_before) < i < len(y_before) + len(y_during):
                    true_during += 1
                elif i >= len(y_before) + len(y_during):
                    true_after += 1

        accuracy['before'] = true_before / len(y_before)*100
        accuracy['during'] = true_during / len(y_during)*100
        accuracy['after'] = true_after / len(y_after)*100
        accuracy['overall'] = (true_after+true_before+true_during) / len(y)*100

        print('Accuracy before the crash: ', accuracy['before'])
        print('Accuracy during crash: ', accuracy['during'])
        print('Accuracy after the crash:', accuracy['after'])
        print('Overall Accuracy: ', accuracy['overall'])

        return accuracy

    def execute(self, all_decisions, actions):
        timestamp = self.environment_observations.index[self.state_iterator]
        self.state_iterator += 1
        reward_val = self.reward_var.iloc[self.state_iterator]
        next_state = self.environment_observations.iloc[self.state_iterator]
        terminal = self.check_terminal_criteria(
            actions, all_decisions, timestamp, early_terminate=False)
        reward = self.calculate_reward(reward_val,
                                       timestamp, actions['DECISION'], terminal)

        return next_state, terminal, reward


class ExchangeAgent:
    def __init__(self, environment: ExchangeEnvironment):

        self.market_evaluation = Agent.create(
            agent='dqn', environment=environment, batch_size=128, memory=len(y))
        self.episode_actions = {}


def execute_episode_batch(environment: ExchangeEnvironment, agent: ExchangeAgent, episodes):
    decision_reward_df = pd.DataFrame()
    for i in range(episodes):
        episode_length = 0
        state_vector = environment.reset()
        all_timestamps = []
        all_decisions = []
        all_rewards = []
        terminate = False
        while not terminate:
            episode_length += 1
            actions = agent.market_evaluation.act(states=state_vector)
            state_vector, terminate, reward = environment.execute(all_decisions,
                                                                  actions=actions)
            agent.market_evaluation.observe(
                terminal=terminate, reward=reward)
            all_decisions.append(actions['DECISION'])
            all_rewards.append(reward)
            all_timestamps.append(state_vector.name)

        if episode_length == len(y):
            decision_reward_df['Timestamp'] = all_timestamps
            decision_reward_df[f'model/Decision_Episode{i}'] = all_decisions
            decision_reward_df[f'Rewards_Episode{i}'] = all_rewards
            accuracy = environment.report_accuracy(all_decisions, y)
            if accuracy['overall'] > 90:
                decision_reward_df.to_excel('Decision_reward.xlsx')

            if accuracy['before'] == 100 and accuracy['during'] > 70 and accuracy['after'] == 100:
                decision_reward_df.to_excel('Decision_reward.xlsx')
                break


def simulator(environment: ExchangeEnvironment, agent: ExchangeAgent, episodes=100):

    execute_episode_batch(environment, agent, episodes)
    environment.close()


if __name__ == "__main__":

    environment_observations = pd.read_csv(
        features_file_path + 'df_environment.csv', index_col='Timestamp')
    exchange_environment = ExchangeEnvironment(
        environment_observations)
    y = environment_observations['LABEL'][1:].to_list()
    exchange_agent = ExchangeAgent(exchange_environment)
    simulator(exchange_environment, exchange_agent)
