#!/usr/bin/env bash
###
# Experiments initializer
###

#
# experiments.sh <experiment_type>
#


# ['evasion','cartpole']
exp=$1

data_type=("kmeans" "optics" "xmeans")
data_seed=(1 2 3)

# shellcheck disable=SC2006
project_base_dir=$(dirname "$0")
publish_dir_base="tmp"

student_config_model="data/evasao/students.json"

evasion() {
  for dtype in "${data_type[@]}"
  do
    for dseed in "${data_seed[@]}"
    do
      echo "Initializing for dataset: ${dtype}/${dseed}"
      docker exec -it busy_gould /bin/bash -c "cd ~/BatchRL/scripts_ufes/ && bash -x dqn_evasao_batch.sh ${dtype} ${dseed} ${publish_dir_base}"
    done
  done
}

visualization(){
  for dtype in "${data_type[@]}"
  do
    local path="${project_base_dir}/${publish_dir_base}/${dtype}"
    # Actions Comparison
    python visualization.py --path "${path}/results" --output "${path}" --plot_all

    # CPE Reward aggregation
    python extract_csv.py --path "${path}/results"
  done
}

fix_permissions(){
  local path="${project_base_dir}/${publish_dir_base}"
  echo "Fixing ${path} permissions..."
  echo "Setting ${path} owner to ${USER}:${USER}"
  sudo chown -R "$USER":"$USER" "$path"
}

generate_evasion_dataset(){
  for dtype in "${data_type[@]}"
  do
    for dseed in "${data_seed[@]}"
    do
      echo "Creating dataset: ${dtype}/${dseed}"
      path="${project_base_dir}/${publish_dir_base}/data/${dtype}"
      mkdir -p "${path}"

      # Fix student config file
      json_student=$(cat ${student_config_model})
      json_student=$(jq '.output_path = $newVal' --arg newVal "${path}" <<< "${json_student}")
      json_student=$(jq '.discrete_state.method = $newVal' --arg newVal "${dtype}" <<< "${json_student}")
      echo "${json_student}" > "${path}/student_${dtype}_${dseed}.json"

      PYTHONPATH=~/PycharmProjects/BatchRL python data/evasao/gen_discrete_student.py -p "${path}/student_${dtype}_${dseed}.json" --seed "${dseed}"

    done
  done

}

case "$exp" in
    "evasion" )
        echo "Initializing experiment $exp"

        echo "Generating dataset"
        generate_evasion_dataset

#        echo "Running experiments"
#        evasion
#
#        echo "Preparing results to analysis"
#        fix_permissions
#        visualization

        echo "Experiment $exp finished"
        ;;
    *)
      echo "Experiment $exp is not valid"
      ;;
esac
