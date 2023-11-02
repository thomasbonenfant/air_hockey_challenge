#!/usr/bin/bash

user=airhockey
server=131.175.120.196
server_root_dir="/home/airhockey/thomas/air_hockey_challenge/"

remote_folder="${user}@${server}:${server_root_dir}"

args=(-avz --exclude "logs" --exclude "downloads" --exclude "air_hockey_agent/agents/Agents" --exclude "__pycache__/" --exclude "mepol/" --exclude "notebooks/" --exclude "tb_logs/" --exclude "docs/" --exclude ".git/")

rsync "${args[@]}" . $remote_folder

