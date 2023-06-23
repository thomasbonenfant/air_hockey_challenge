python -m mepol.src.experiments.mepol \
    --exp_name "Test Acceleration Space" \
    --env "AirHockeyAcc" \
    --env_name "3dof-hit" \
    --k 5 \
    --kl_threshold 0.1 \
    --max_off_iters 0 \
    --learning_rate 0.00001 \
    --num_trajectories 1 \
    --trajectory_length 400 \
    --num_epochs 10000 \
    --flag_heatmap 0 \
    --heatmap_every 100 \
    --heatmap_episodes 100 \
    --heatmap_num_steps 400 \
    --use_backtracking 1 \
    --zero_mean_start 1 \
    --full_entropy_traj_scale 5 \
    --full_entropy_k 5 \
    --num_workers 1 \
    --task_space 1 \
    --task_space_vel 0 \
    --use_delta_pos 0 \
    --delta_dim 1 \
    --use_puck_distance 1 \
    --use_tanh 1 \
    --max_acc 5 \
    --log_std -0.5 \
    -s 0 -s 1 -s 10 -s 11 \



