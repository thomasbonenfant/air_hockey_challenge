python -m my_scripts.experiment \
\
--total_timesteps=1000000 \
--save_model_dir="/tmp/" \
--experiment_label="rb_hit/defend_oac/repel_oac/prepare_rb" \
\
--steps_per_action=15 \
--include_timer \
--scale_obs \
--alpha_r=0.0 \
--large_reward=1000 \
--fault_penalty=1000 \
--fault_risk_penalty=0 \
--parallel=2 \
\
--eval_freq=2048 \
--n_eval_episodes=10 \
\
dqn \
--learning_rate=0.001 \
--buffer_size=10000 \
--learning_starts=1000 \
--batch_size=32 \
--tau=1.0 \
--gamma=0.99 \
--target_update_interval=100 \
--exploration_fraction=0.1 \
--exploration_initial_eps=1.0 \
--exploration_final_eps=0.02 \
--max_grad_norm=10.0 \
--train_freq=1 \
--gradient_steps=1 \
--stats_window_size=100 \
--_init_setup_model
