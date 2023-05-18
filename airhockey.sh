python -m mepol.src.experiments.mepol --exp_name "Task Space No Delta 2nd try" --env "AirHockey" \
    --k 3000  --kl_threshold 0 --max_off_iters 0 --learning_rate 0.00001 \
    --num_trajectories 20 --trajectory_length 400 --num_epochs 10000 --heatmap_every 200 \
    --heatmap_episodes 100 --heatmap_num_steps 400 --use_backtracking 1 --zero_mean_start 1 \
    --full_entropy_traj_scale 2 --full_entropy_k 3000 --num_workers 2 \
    --task_space 1 --task_space_vel 0 --use_delta_pos 0 --use_puck_distance 1 --use_tanh=0 --seed 44299
