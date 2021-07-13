# Discovering an Aid Policy to Minimize Student Evasion Using Offline Reinforcement Learning
Offline Reinforcement Learning

Algorithms:  
- DQN 

Evaluation:  
- CFE  
- Reward  
  
Environments:  
- [Cartpole](#cartpole)  
- [Student dropout](#student-dropout) 
  
## Environments 
Current environments.

### Cartpole 
Classic cart-pole environment.

### Student dropout  
Select aid actions to minimize student dropout.
  
## Configurations
Required settings.

### Data Configuration
Generate discrete data.
  
Clustering algorithm options:
- k-means  
- x-means  
- optics  
  
Execute:  

	# python gen_discrete_student.py -p <config_file.json> --seed=N 
	python gen_discrete_student.py -p students.json --seed=3 

Settings example in `scripts_ufes/data/students.json` file. In the example file there are:  
 - "data_path": "/path/full_data.pkl"  
    - Original data file path in pkl format.
 - "output_path": "outputs/output_path/"  
    - Path where all state discretization script outputs will be saved. Including PCA visualization of discretization, except for optics (cannot generate it).

The visualization via PCA of the discretized data is found in `<output_path>/pca_data_discretization.png`  
 
### Test Configuration
In `dqn_evasao_batch.sh` file, there are some settings that need to be defined:

| Variable | Default | Descrição |
|--|--|--|
|`DATASET`| "xmeans/1"  | |
|`test_name`| "evasao"   |  |
|`test_basedir`| "/tmp"  |  |
|`project_base_dir`| "/home/BatchRL" |  |
|`project_dir`| "${project_base_dir}/ReAgent/" |  |
|`python_prefix`| "PYTHONPATH=/home/BatchRL/ReAgent" | |
|`publish_dir`| "/home/BatchRL/scripts_ufes/tmp"  ||
|`seed`| 0 | |
|`softmax_temperature`| "0.35"| |
|`policy_data_gen`| "${project_dir}/ml/rl/test/gym/discrete_dqn_cartpole_v0.json"| |
|`timeline_config`|"${project_base_dir}/scripts_ufes/dqn_evasao_batch/workflow/timeline.json"||
|`workflow_config`| "${project_base_dir}/scripts_ufes/dqn_evasao_batch/workflow/dqn_example.json" ||
|`fixed_data_path`| "/tmp/fixo/data"||

## Tests  
  
Execute the following command to run a test:
  
	 # docker exec -it <container_ID> /bin/bash -c "cd ~/BatchRL/scripts_ufes/ && bash -x dqn_evasao_batch.sh <DATASET>"
	 docker exec -it busy_gould /bin/bash -c "cd ~/BatchRL/scripts_ufes/ && bash -x dqn_evasao_batch.sh xmeans/1" 
 
## Results  
  
The generated artifacts will be saved in `scripts_ufes/tmp/<environment>_<test_ID>`.  
  
The results will be saved in `scripts_ufes/tmp/<environment>_<test_ID>/outputs/`.  
  
To view the results of all runs, start the tensorboard:  
  
 tensorboard --logdir scripts_ufes/tmp

For the plot comparing between the actions of the database and the actions of the learned policy, currently, the input is coded in the script and saves the output in`/tmp/actions_comparison_bar.png`.
 
    # Todos os modelos em <PATH> juntos e média dos logados
    # Plot all models in <PATH> and mean logged actions
    # python scripts_ufes/visualization.py --path <PATH> --output <OUTPUT_PATH> --plot_all
    python scripts_ufes/visualization.py --path /tmp/docs/kmeans/result --output /tmp/docs/kmeans --plot_all
    # Plot only one model
    python scripts_ufes/visualization.py --path /tmp/docs/kmeans/result/1 --output /tmp/docs/kmeans/1
    
To generate a plot with CPE values:

    python scripts_ufes/extract_csv.py --path="<publish_dir>"
    
## Citation
If you find our project useful in your research, please consider citing:
```
@misc{delima2021discovering,
      title={Discovering an Aid Policy to Minimize Student Evasion Using Offline Reinforcement Learning}, 
      author={Leandro M. de Lima and Renato A. Krohling},
      year={2021},
      eprint={2104.10258},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
