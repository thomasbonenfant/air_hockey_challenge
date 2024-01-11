#!/bin/bash

#Source directory on the remote server
remote_directory="airhockey@131.175.120.196:/home/airhockey/thomas/data"

#Local destination
local_directory="/home/thomas/Downloads/markov"

#Filter rules
filter_rules=(
  "+ model.zip"
  "+ best_model.zip"
  "+ config.yaml"
  "- *.csv"
  "+ variant.json"
  "- checkpoints/"
  "- tb_logs*/"
  "- *.pkl"
  "- air_hockey/"
  "- *.zip_pkl"
  "- *.log"
)

filter_string=""

for rule in "${filter_rules[@]}"
do
  filter_string+="-f'$rule' "
done

rsync_command="rsync -avz --progress $filter_string $remote_directory $local_directory"
eval "$rsync_command"