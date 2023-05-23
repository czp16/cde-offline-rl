# Learning from Sparse Offline Datasets via Conservative Density Estimation

This project provides the open source implementation of the CDE in the paper: "Learning from Sparse Offline Datasets via Conservative Density Estimation"

## Installation
1) We recommend to use Anaconda or Miniconda to manage python environment.
2) Install `mujoco` and `mujoco-py`, your can either refer to https://github.com/openai/mujoco-py or run 
    ```shell
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    tar -xvf mujoco210-linux-x86_64.tar.gz
    mkdir .mujoco
    mv mujoco210 ~/.mujoco/mujoco210
    ```
    It is also necessary to add below to `~/.bashrc`:
    ```shell
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    ```
    We have included `mujoco-py` in `requirements.txt` but you may need to install `libglew-dev, patchelf` when compiling the `mujoco-py` after the installation:
    ```shell
    sudo apt-get install libglew-dev
    sudo apt-get install patchelf
    ```
3) Create conda env:
    ```shell
    cd cde-offline-rl
    conda env create -f environment.yaml
    conda activate cde
    ```
4) Install PyTorch according to your platform and cuda version.
5) Install D4rl from https://github.com/Farama-Foundation/D4RL.

## Training
To run a single experiment, take maze2d-medium-v1 for example, run
```python
python run_cde.py --env_name "maze2d-medium-v1" --hyperparams 'hyper_params/cde/maze2d.yaml' --cudaid 0 --seed 100
```
where `--hyperparams` specifies the hyperparameter files in directory `./hyper_params/`, `--cudaid` specifies which gpu will be used for training (the defaulted `-1` means using cpu).

If you want to run multiple experiments, we have also included some other training commands in bash file `run_exp.sh`. You can consider using `&` to run the commands in parallel.