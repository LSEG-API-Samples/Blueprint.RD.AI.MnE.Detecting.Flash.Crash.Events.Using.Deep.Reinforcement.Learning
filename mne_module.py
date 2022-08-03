from tensorforce import Environment
from reinforcement_learning_module import ExchangeEnvironment, ExchangeAgent


features_file_path = './studies/flash_crashes/omx30/features/'


def modelling_and_evaluation_phase():
    exchange_environment = Environment.create(environment=ExchangeEnvironment, max_episode_timesteps=1000)
    exchange_agent = ExchangeAgent(exchange_environment)

    episodes = 100
    for _ in range(episodes):
        states = exchange_environment.reset()
        terminal = False
        while not terminal:
            actions = exchange_agent.act(states)
            states, terminal, reward = exchange_environment.execute(actions=actions)
            exchange_agent.observe(terminal=terminal, reward=reward)