python -m mepol.src.experiments.mepol --exp_name "Randomized Puck, Delta Action" --env "AirHockey" \
    --k 1000 \
    --kl_threshold 0.1 \
    --max_off_iters 30 \
    --learning_rate 0.00004 \
    --num_trajectories 20 \
    --trajectory_length 400 \
    --num_epochs 10000 \
    --flag_heatmap 0 \
    --heatmap_every 200 \
    --heatmap_episodes 100 \
    --heatmap_num_steps 400 \
    --use_backtracking 1 \
    --zero_mean_start 1 \
    --full_entropy_traj_scale 1 \
    --full_entropy_k 1000 \
    --num_workers 20 \
    --task_space 1 \
    --scale_task_space_action 1 \
    --task_space_vel 0 \
    --use_delta_pos 1 \
    --delta_dim 0.1 \
    --use_puck_distance 1 \
    --normalize_obs 1 \
    --use_tanh 1 \
    --log_std -1.6 \
    -s 0 -s 1 -s 10 -s 11 \


