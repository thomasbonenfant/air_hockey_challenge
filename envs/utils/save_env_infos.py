from air_hockey_challenge.framework import AirHockeyChallengeWrapper
import pickle

env_hit = AirHockeyChallengeWrapper("7dof-hit", interpolation_order=3)
env_defend = AirHockeyChallengeWrapper("7dof-defend", interpolation_order=3)

env_info_hit = env_hit.env_info
env_info_defend = env_defend.env_info

with open("env_info_single_agent/env_infos.pkl", "wb") as fp:
    pickle.dump([env_info_hit, env_info_defend], fp)

