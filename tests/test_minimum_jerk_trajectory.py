
from air_hockey_challenge.framework.custom_environment_wrapper import CustomEnvironmentWrapper
from air_hockey_challenge.utils.kinematics import forward_kinematics
from air_hockey_challenge.utils.transformations import robot_to_world
from utils.trajectory_logger import TrajectoryLogger
from air_hockey_agent.agents.dummy_agent import DummyAgent
from air_hockey_agent.agents.defend_agent import SimpleDefendingAgent

if __name__ == '__main__':

    env = CustomEnvironmentWrapper(env="3dof-defend")

    agent = SimpleDefendingAgent(env.base_env.env_info,steps_per_action=50)

    logger = TrajectoryLogger()

    obs = env.reset()
    agent.reset()

    steps = 0
    while True:
        steps +=1 
        action = agent.draw_action(obs)
        obs, reward, done, info = env.step(action)
        #env.render()

        puck_pos = obs[env.base_env.env_info['puck_pos_ids']][:2]
        joint_pos = obs[env.base_env.env_info['joint_pos_ids']]

        mj_model = env.base_env.env_info['robot']['robot_model']
        mj_data = env.base_env.env_info['robot']['robot_data']

        ee_pos = forward_kinematics(mj_model, mj_data, joint_pos)[0]

        ee_pos = robot_to_world(env.base_env.env_info['robot']['base_frame'][0], ee_pos)[0][:2]
        puck_pos = robot_to_world(env.base_env.env_info['robot']['base_frame'][0], puck_pos)[0][:2]

        logger.append2trajectory(ee_pos, puck_pos, action)
        
        if done or steps > env.info.horizon / 2:
         break

    logger.visualize(env.env_info)
    logger.plot_coordinate_traj()
