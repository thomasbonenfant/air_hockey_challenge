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
--parallel=15 \
\
--eval_freq=2048 \
--n_eval_episodes=10 \
\
ppo \
--lr=3e-4 \
--steps_per_update=256 \
--batch_size=256 \
--n_epochs=20 \
--gamma=0.997 \
--gae_lambda=0.95 \
--clip_range=0.2 \
--ent_coef=0.05 \
--vf_coef=0.5 \
--max_grad_norm=0.5 \
--sde_sample_freq=-1 \
--stats_window_size=100 \
--tensorboard_log="./" \
--verbose=1 \







