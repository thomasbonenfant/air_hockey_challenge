python -m mepol.src.experiments.mepol \
    --exp_name "Only puck after best of joint entropy" \
    --env "AirHockey" \
    --env_name "3dof-hit" \
    --k 500 \
    --kl_threshold 0.0001 \
    --max_off_iters 0 \
    --learning_rate 0.00001 \
    --num_trajectories 20 \
    --trajectory_length 400 \
    --num_epochs 10000 \
    --flag_heatmap 0 \
    --heatmap_every 100 \
    --heatmap_episodes 100 \
    --heatmap_num_steps 400 \
    --use_backtracking 0 \
    --zero_mean_start 1 \
    --full_entropy_traj_scale 5 \
    --full_entropy_k 500 \
    --num_workers 2 \
    --task_space 1 \
    --task_space_vel 0 \
    --use_delta_pos 1 \
    --delta_dim 0.4 \
    --use_puck_distance 1 \
    --use_tanh 1 \
    --log_std -0.5 \
    -s 0 -s 1 \
    --policy_init "/home/thomas/PycharmProjects/air_hockey_challenge/policies/best_40traj_KL_low-policy" \
    --log_dir mepol/results



