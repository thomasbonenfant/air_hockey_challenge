python -m my_scripts.experiment \
\
--total_timesteps=1000000 \
--save_model_dir="/home/airhockey/thomas/data/hit" \
--experiment_label="joints+ee_pos+ee_vel" \
\
--env=hit \
--scale_obs \
--scale_action \
--alpha_r=10.0 \
--parallel=1 \
--include_ee \
--include_ee_vel \
--include_joints \
\
--eval_freq=2048 \
--n_eval_episodes=100 \
\
sac \
--learning_rate=3e-4 \
--buffer_size=1000000 \
--learning_starts=1000 \
--batch_size=256 \
--tau=0.005 \
--gamma=0.997 \
--train_freq=1 \
--gradient_steps=1 \
--ent_coef=auto \
--target_update_interval=1 \
--target_entropy=auto \
--sde_sample_freq=-1 \
--stats_window_size=100 \
--tensorboard_log="./" \
--verbose=1 \
--device=auto \
--_init_setup_model
