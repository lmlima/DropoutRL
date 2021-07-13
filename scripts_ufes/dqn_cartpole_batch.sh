#!/usr/bin/env bash
# git submodule update --init --recursive
# git submodule update --recursive
# Prepare docker with cuda: docker build -f docker/cuda.Dockerfile -t horizon:dev --memory=8g --memory-swap=8g .
# Create docker with cuda: docker run --runtime=nvidia -v $PWD:/home/BatchRL -p 0.0.0.0:6006:6006 -it horizon:dev
# TODO: Add below preparation steps in a new dockerfile
# Inside docker: sudo apt install jq moreutils -y
# In $project_dir: mvn -f preprocessing/pom.xml clean package
# ./scripts/setup.sh
###
# Configuration
###
test_name="cartpole_batch"
test_basedir="/tmp"
project_base_dir="/home/BatchRL"
project_dir="${project_base_dir}/ReAgent/"
python_prefix="PYTHONPATH=/home/BatchRL/ReAgent"
publish_dir="/home/BatchRL/scripts_ufes/tmp"

seed=0
softmax_temperature="0.35"
policy_data_gen="${project_dir}/ml/rl/test/gym/discrete_dqn_cartpole_v0.json"

timeline_config="${project_base_dir}/scripts_ufes/dqn_cartpole_batch/workflow/timeline.json"
workflow_config="${project_base_dir}/scripts_ufes/dqn_cartpole_batch/workflow/dqn_example.json"
##########
fixed_data_path="/tmp/fixo/data"

generate_data() {
  ###
  # Create training data
  ###
  echo "Generating new data..."

  policy="$policy_data_gen"
  output_file="${test_dir}/${test_name}/training_data.json"

  python "${project_dir}"/ml/rl/test/gym/run_gym.py -p "$policy" -f "${output_file}" --seed "$seed"
  # shellcheck disable=SC2002
  cat "$output_file" | head -n1 | python -m json.tool
}

convert_data_to_timeline() {
  ### Convert data to timeline format ###
  echo "Converting data to timeline..."

  # Build timeline package (only need to do this first time)
  # mvn -f preprocessing/pom.xml clean package

  # Clear last run's spark data (in case of interruption)
#  rm -Rf spark-warehouse derby.log metastore_db preprocessing/spark-warehouse preprocessing/metastore_db preprocessing/derby.log
  cat "${test_dir}/timeline.json"
  timeline_content=$(cat "${test_dir}/timeline.json")

  # Run timelime on pre-timeline data
  (cd "${test_dir}" && \
    rm -Rf spark-warehouse derby.log metastore_db preprocessing/spark-warehouse preprocessing/metastore_db preprocessing/derby.log && \
    /usr/local/spark/bin/spark-submit \
    --class com.facebook.spark.rl.Preprocessor "${project_dir}"preprocessing/target/rl-preprocessing-1.1.jar \
    "$timeline_content" )

  # Merge output data to single file
  cat "${test_dir}/${test_name}_tmp_training"/part* > "${training_data_dir}/${test_name}_timeline.json"
  cat "${test_dir}/${test_name}_tmp_eval"/part* > "${training_data_dir}/${test_name}_timeline_eval.json"

  # Remove the output data folder
  rm -Rf "${test_dir}/${test_name}_tmp_training" "${test_dir}/${test_name}_tmp_eval"
}

create_metadata(){
  ### Create normalization parameters ###
  echo "Creating normalization metadata..."

  python "${project_dir}"ml/rl/workflow/create_normalization_metadata.py -p "${test_dir}/workflow.json"
  python -m json.tool <<< cat "${training_data_dir}/state_features_norm.json"
}

train_policy(){
  python "${project_dir}"ml/rl/workflow/dqn_workflow.py -p "${test_dir}/workflow.json" 2>&1 | tee "${output_dir}/policy_training.log"
}

evaluate_policy(){
  python "${project_dir}"ml/rl/test/workflow/eval_cartpole.py -m outputs/model_* --softmax_temperature="${softmax_temperature}" --log_file="${output_dir}/eval_output.txt"
}

generate_config_files(){

  # Fix timeline config file
  json_timeline=$(cat ${timeline_config})
  json_timeline=$(jq '.timeline.inputTableName = $newVal' --arg newVal "${test_name}" <<< "${json_timeline}")
  json_timeline=$(jq '.timeline.outputTableName = $newVal' --arg newVal "${test_name}_tmp_training" <<< "${json_timeline}")
  json_timeline=$(jq '.timeline.evalTableName = $newVal' --arg newVal "${test_name}_tmp_eval" <<< "${json_timeline}")
  echo "${json_timeline}" > "${test_dir}/timeline.json"

  # Fix paths in workflow config file
  json_workflow=$(cat "${workflow_config}")
  json_workflow=$(jq '.training_data_path = $newVal' --arg newVal "${training_data_dir}/${test_name}_timeline.json" <<< "${json_workflow}")
  json_workflow=$(jq '.eval_data_path = $newVal' --arg newVal "${training_data_dir}/${test_name}_timeline_eval.json" <<< "${json_workflow}")
  json_workflow=$(jq '.state_norm_data_path = $newVal' --arg newVal "${training_data_dir}/state_features_norm.json" <<< "${json_workflow}")
  json_workflow=$(jq '.model_output_path = $newVal' --arg newVal "${output_dir}/" <<< "${json_workflow}")
  json_workflow=$(jq '.norm_params.output_dir = $newVal' --arg newVal "${training_data_dir}/" <<< "${json_workflow}")
  echo "${json_workflow}" > "${test_dir}/workflow.json"

}
###
###
###
# Generate paths
timestamp=$(date +%s)
test_dir="${test_basedir}/${test_name}_${timestamp}"
training_data_dir="${test_dir}/training_data"
output_dir="${test_dir}/outputs"

mkdir -p "${training_data_dir}"
mkdir -p "${test_dir}/${test_name}"
mkdir -p "${output_dir}"


export ${python_prefix}

generate_config_files | ts

#cp "${fixed_data_path}/training_data.json" "${test_dir}/${test_name}/training_data.json"
generate_data | ts

convert_data_to_timeline | ts

create_metadata | ts

### Train ###
train_policy | ts

#### Eval ###
#evaluate_policy | ts

# Publish results
mkdir -p "${publish_dir}/${test_name}_${timestamp}"
cp -Rf "${output_dir}" "${publish_dir}/${test_name}_${timestamp}"
cp -Rf "${training_data_dir}" "${publish_dir}/${test_name}_${timestamp}"
echo "Output published in: ${publish_dir}/${test_name}_${timestamp}"
echo "Run: tensorboard --logdir ${publish_dir}"

# Clean
#rm -Rf /tmp/cartpole_batch_*
