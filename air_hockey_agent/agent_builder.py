from air_hockey_challenge.framework.agent_base import AgentBase
from baseline.baseline_agent.tactics import *
import keras as tfk

def build_agent(env_info, **kwargs):
    """
    Function where an Agent that controls the environments should be returned.
    The Agent should inherit from the mushroom_rl Agent base env.

    :param env_info: The environment information
    :return: Either Agent ot Policy
    """
    if "hit" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1)
    if "defend" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="defend")
    if "prepare" in env_info["env_name"]:
        return BaselineAgent(env_info, **kwargs, agent_id=1, only_tactic="prepare")

    return BaselineAgent(env_info, **kwargs)


class BaselineAgent(AgentBase):
    def __init__(self, env_info, agent_id=1, only_tactic=None, **kwargs):
        super(BaselineAgent, self).__init__(env_info, agent_id, **kwargs)

        if self.env_info['robot']['n_joints'] == 3:
            joint_anchor_pos = np.array([-1.15570723, 1.30024401, 1.44280414])
            x_init = np.array([0.65, 0., 0.1])
            x_home = np.array([0.65, 0., 0.1])
            max_hit_velocity = 1.0
        else:
            joint_anchor_pos = np.array([0., -0.1961, 0., -1.8436, 0., 0.9704, 0.])
            x_init = np.array([0.65, 0., self.env_info['robot']['ee_desired_height'] + 0.2])
            x_home = np.array([0.65, 0., self.env_info['robot']['ee_desired_height']])
            max_hit_velocity = 1.2

        self.agent_params = {'switch_tactics_min_steps': 15,
                             'max_prediction_time': 1.0,
                             'max_plan_steps': 5,
                             'static_vel_threshold': 0.15,
                             'transversal_vel_threshold': 0.1,
                             'joint_anchor_pos': joint_anchor_pos,
                             'default_linear_vel': 0.6,
                             'x_init': x_init,
                             'x_home': x_home,
                             'hit_range': [0.8, 1.3],
                             'max_hit_velocity': max_hit_velocity,
                             'defend_range': [0.8, 1.0],
                             'defend_width': 0.45,
                             'prepare_range': [0.8, 1.3]}

        self.state = SystemState(self.env_info, agent_id, self.agent_params)
        self.traj_generator = TrajectoryGenerator(self.env_info, self.agent_params, self.state)

        self.tactics_processor = [Init(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  Ready(self.env_info, self.agent_params, self.state, self.traj_generator,
                                        only_tactic=only_tactic),
                                  Prepare(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  Defend(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  Repel(self.env_info, self.agent_params, self.state, self.traj_generator),
                                  Smash(self.env_info, self.agent_params, self.state, self.traj_generator)]

        self.lstm = tfk.models.load_model("LSTMwithScale")

        self.min = np.array([0.0, -1.0, -3.14159265, -20.0, -20.0, -100.0])
        self.max = np.array([3.02, 1.0, 3.14159265, 20.0, 20.0, 100.0])

        self.sequence_length = 10
        # 6 is the x,y and yaw of pos and vel
        # self.last_seq = np.zeros(1, self.sequence_length, 6)
        self.last_seq = np.empty((1, self.sequence_length, 6))
        self.initial = True

    def preprocess(self, puckpos, puckvel):
        puckstate = np.concatenate((puckpos, puckvel), axis=0).reshape(1, 1, 6)
        puckstate = (puckstate - self.min) / (self.max - self.min)


        if self.initial:
            self.last_seq = np.tile(puckstate, (1, self.sequence_length, 1))
            self.initial = False
        else:
            self.last_seq = np.roll(self.last_seq, shift=-1, axis=1)
            self.last_seq[:, -1, :] = puckstate
        pred = self.lstm.predict([self.last_seq, puckstate], verbose=0)
        # zeros = np.zeros((pred.shape[0], pred.shape[1], 4))
        zeros = np.zeros((pred.shape[0], pred.shape[1], 3))

        result = np.concatenate((pred, zeros), axis=-1)
        denoised = (result * (self.max - self.min)) + self.min
        puckposx = denoised[0, 0, 0]
        puckposy = denoised[0, 0, 1]
        puckposyaw = denoised[0, 0, 2]

        return [puckposx, puckposy, puckposyaw]

    def reset(self):
        self.state.reset()
        self.initial = True




    def draw_action(self, obs):
        puck_pos = self.get_puck_pos(obs)
        puck_vel = self.get_puck_vel(obs)

        # denoiser with only pos x and y as output
        # puck_new_posx, puck_new_posy = self.preprocess(puck_pos, puck_vel)
        # self.state.update_observation(self.get_joint_pos(obs), self.get_joint_vel(obs), [puck_new_posx,
        #                                                                                  puck_new_posy,
        #                                                                                  self.get_puck_pos(obs)[2]])

        # denoiser with x, y and yaw as output
        # puck_state = self.preprocess(puck_pos, puck_vel)
        # self.state.update_observation(self.get_joint_pos(obs), self.get_joint_vel(obs), puck_state)

        # without denoiser
        self.state.update_observation(self.get_joint_pos(obs), self.get_joint_vel(obs), self.get_puck_pos(obs))

        while True:
            self.tactics_processor[self.state.tactic_current.value].update_tactic()
            activeTactic = self.tactics_processor[self.state.tactic_current.value]

            if activeTactic.ready():
                activeTactic.apply()
            if len(self.state.trajectory_buffer) > 0:
                break
            else:
                # print("iterate")
                pass

        self.state.q_cmd, self.state.dq_cmd = self.state.trajectory_buffer[0]
        self.state.trajectory_buffer = self.state.trajectory_buffer[1:]

        self.state.x_cmd, self.state.v_cmd = self.state.update_ee_pos_vel(self.state.q_cmd, self.state.dq_cmd)
        return np.vstack([self.state.q_cmd, self.state.dq_cmd])


def main():
    import time
    from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
    np.random.seed(0)

    env = AirHockeyChallengeWrapper(env="3dof-hit-opponent", action_type="position-velocity",
                                    interpolation_order=3, debug=False)

    agents = BaselineAgent(env.env_info, agent_id=1, only_tactic="hit")
    # agent2 = BaselineAgent(env.env_info, agent_id=1)

    # agents = DoubleAgentsWrapper(env.env_info, agent1, agent2)

    obs = env.reset()
    agents.episode_start()

    steps = 0
    while True:
        steps += 1
        t_start = time.time()
        action = agents.draw_action(obs)
        # print("time: ", time.time() - t_start)
        obs, reward, done, info = env.step(action)

        env.render()

        if done or steps > env.info.horizon:
            # import matplotlib.pyplot as plt
            # import matplotlib
            # matplotlib.use("tkAgg")
            # trajectory_record = np.array(env.base_env.controller_record)
            # nq = env.base_env.env_info['robot']['n_joints']
            #
            # fig, axes = plt.subplots(3, nq)
            # for j in range(nq):
            #     axes[0, j].plot(trajectory_record[:, j])
            #     axes[0, j].plot(trajectory_record[:, j + nq])
            #     axes[1, j].plot(trajectory_record[:, j + 2 * nq])
            #     axes[1, j].plot(trajectory_record[:, j + 3 * nq])
            #     # axes[2, j].plot(trajectory_record[:, j + 4 * nq])
            #     axes[2, j].plot(trajectory_record[:, j + nq] - trajectory_record[:, j])
            # plt.show()

            steps = 0
            obs = env.reset()
            agents.episode_start()
            print("Reset")


if __name__ == '__main__':
    main()