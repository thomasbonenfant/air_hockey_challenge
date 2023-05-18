#!/usr/bin/bash

user=air_hockey
server=turing.deib.polimi.it
server_root_dir="thomas/air_hockey_challenge/"
mepol_src="mepol/src/"
air_hockey_challenge_dir="air_hockey_challenge/"
air_hockey_agent_dir="air_hockey_agent/"

remote_folder="${user}@${server}:${server_root_dir}"

args=(-avz --exclude "__pycache__/")

rsync "${args[@]}"  $mepol_src $remote_folder$mepol_src

rsync "${args[@]}" $air_hockey_challenge_dir $remote_folder$air_hockey_challenge_dir

rsync "${args[@]}" $air_hockey_agent_dir $remote_folder$air_hockey_agent_dir

rsync -avz airhockey.sh "$remote_folder"airhockey.sh
