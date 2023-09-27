# ====== maze2d =======
env_names=("maze2d-large" "maze2d-medium" "maze2d-umaze")

for env in ${env_names[@]};
do 
    python run_cde.py --env_name "${env}-v1" --hyperparams 'hyper_params/cde/maze2d.yaml' --cudaid 0 --seed 100 
done

# ====== mujoco =======
env_names=("halfcheetah" "hopper" "walker2d")
levels=("medium-expert" "medium") # 

for lvl in ${levels[@]};
do
    for env in ${env_names[@]};
    do
        python run_cde.py --env_name "${env}-${lvl}-v2" --hyperparams 'hyper_params/cde/mujoco.yaml' --cudaid 0 --seed 100
    done
done