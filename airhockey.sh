python -m mepol.src.experiments.mepol --env "AirHockey" \
    --k 4  --kl_threshold 15 --max_off_iters 30 --learning_rate 0.00001 \
    --num_trajectories 20 --trajectory_length 400 --num_epochs 10000 --heatmap_every 25 \
    --heatmap_episodes 100 --heatmap_num_steps 400 --use_backtracking 1 --zero_mean_start 1 \
    --full_entropy_traj_scale 5 --full_entropy_k 4 --num_workers 1