import yaml
from hemac import HeMAC_v0
import tqdm

def test_hemac_random():
    with open("hemac_env_config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)    

    # config["render_mode"] = "human"
    config["area_size"] = [500, 500]  # Smaller arena for testing
    env = HeMAC_v0.env(**config)
    env.reset(seed=0)

    rewards = 0
    episodes = 1000

    for i in tqdm.trange(episodes):  # Run 5 episodes
        env.reset()
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            if reward != 0:
                # print(f"Agent: {agent}, Reward: {reward}")
                rewards += reward
            if termination or truncation:
                action = None
            else:
                # this is where you would insert your policy
                action = env.action_space(agent).sample()
            env.step(action)
    env.close()

    print(f"Average rewards after {episodes} episodes: {rewards/episodes}")

if __name__ == "__main__":
    test_hemac_random()