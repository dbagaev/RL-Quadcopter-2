from tasks import Takeoff, Hover
from agents import DeepDPGAgent

from helpers import learn_agent

# Setup learning parameters
DeepDPGAgent.tau = 0.01
DeepDPGAgent.gamma = 0.99
DeepDPGAgent.learning_rate = 0.0001
DeepDPGAgent.batch_size = 64

# Create task and agent
# task = Takeoff()
task = Hover(simplified=True)
task.task_name = 'emulate_' + task.task_name

agent = DeepDPGAgent(task, batch_size=128)

num_episodes = 2000


learn_agent(20, agent, task, log_episode_every=10)