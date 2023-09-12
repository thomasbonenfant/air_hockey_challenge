python -m my_scripts.experiment \
--seed=883813 \
--save_model_dir="models" \
--experiment_label="hit_policy_rb+defend_oac+constr" \
--alg="ppo" \
--steps_per_action= \
--include_timer \
--include_faults \
--scale_obs \
--alpha_r=1 \
--large_reward=100 \
--fault_penalty=33.33 \
--fault_risk_penalty=0.1 \
--parallel=8 \
--lr=3e-4 \
--steps_per_update=256 \
--batch_size=64 \
--n_epochs=10 \
--gamma=0.999 \
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
--eval_freq=20480 \
--n_eval_episodes=50 \







