python -m my_scripts.experiment \
\
--total_timesteps=4000000 \
--save_model_dir="/home/airhockey/thomas/data/hit" \
--experiment_label="joints+ee_pos+ee_vel" \
\
--eval_freq=2048 \
--n_eval_episodes=1000 \
\
--env=hit \
--include_joints \
--include_ee \
--include_ee_vel \
--scale_obs \
--scale_action \
--alpha_r=10.0 \
--parallel=20 \
--hit_coeff=1000 \
--max_path_len=400 \
\
ppo \
--lr=3e-4 \
--steps_per_update=256 \
--batch_size=256 \
--n_epochs=20 \
--gamma=0.997 \
--gae_lambda=0.95 \
--clip_range=0.2 \
--ent_coef=0.001 \
--vf_coef=0.5 \
--max_grad_norm=0.5 \
--sde_sample_freq=-1 \
--stats_window_size=100 \
--tensorboard_log="./" \
--verbose=1 \








