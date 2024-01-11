import numpy as np


def vectorize(reward_func):
    def vectorized_reward_fun(env, achieved_goals, desired_goals, infos):
        if achieved_goals.ndim > 1:
            reward_arr = []
            info_arr = []
            gen = (reward_func(env, ach, des, i) for ach, des, i in zip(achieved_goals, desired_goals, infos))
            for r, i in gen:
                reward_arr.append(r)
                info_arr.append(i)

            return np.vstack(reward_arr), np.array(info_arr)
        return reward_func(env, achieved_goals, desired_goals, infos)
    return vectorized_reward_fun

