python -m mepol.src.experiments.mepol \
    --exp_name "Retry Downward Trend Entropy" \
    --env "AirHockey" \
    --k 500 \
    --kl_threshold 0.2 \
    --max_off_iters 30 \
    --learning_rate 0.00001 \
    --num_trajectories 20 \
    --trajectory_length 400 \
    --num_epochs 10000 \
    --flag_heatmap 0 \
    --heatmap_every 100 \
    --heatmap_episodes 100 \
    --heatmap_num_steps 400 \
    --use_backtracking 1 \
    --zero_mean_start 1 \
    --full_entropy_traj_scale 1 \
    --full_entropy_k 500 \
    --num_workers 10 \
    --task_space 1 \
    --scale_task_space_action 0 \
    --task_space_vel 0 \
    --use_delta_pos 0 \
    --delta_dim 0.1 \
    --use_puck_distance 1 \
    --normalize_obs 0 \
    --use_tanh 1 \
    --log_std -2.3 \
    -s 0 -s 1 -s 10 -s 11 \



