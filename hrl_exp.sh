python -m my_scripts.experiment \
--save_model_dir="models" \
--experiment_label="hit_rb+hit_oac+defend/no_constr" \
--alg="ppo" \
--steps_per_action=25 \
--include_timer \
--include_faults \
--scale_obs \
--alpha_r=0.0 \
--large_reward=1000 \
--fault_penalty=333 \
--fault_risk_penalty=1 \
--parallel=8 \
--lr=3e-4 \
--steps_per_update=256 \
--batch_size=256 \
--n_epochs=100 \
--gamma=0.997 \
--gae_lambda=0.95 \
--clip_range=0.2 \
--ent_coef=0.0 \
--vf_coef=0.5 \
--max_grad_norm=0.5 \
--sde_sample_freq=-1 \
--stats_window_size=100 \
--tensorboard_log="./" \
--verbose=1 \
--total_timesteps=1000000 \
--eval_freq=2048 \
--n_eval_episodes=10 \







