python -m my_scripts.experiment \
\
--total_timesteps=1000000 \
--save_model_dir="/home/airhockey/thomas/data/hit" \
--experiment_label="" \
\
--env=hit \
--scale_obs \
--alpha_r=0.0 \
--parallel=1 \
--include_ee \
--include_ee_vel \
--include_joints \
\
--eval_freq=2048 \
--n_eval_episodes=10 \
\
sac \
--learning_rate=3e-4 \
--buffer_size=1000000 \
--learning_starts=1000 \
--batch_size=256 \
--tau=0.005 \
--gamma=0.99 \
--train_freq=1 \
--gradient_steps=1 \
--optimize_memory_usage=false \
--ent_coef=auto \
--target_update_interval=1 \
--target_entropy=auto \
--use_sde=false \
--sde_sample_freq=-1 \
--use_sde_at_warmup=false \
--stats_window_size=100 \
--tensorboard_log="./" \
--verbose=1 \
--device=auto \
--_init_setup_model=true
