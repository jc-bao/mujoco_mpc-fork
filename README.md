# Feedback Sampling with Mujoco MPC

## How to install

```
sudo apt-get install libstdc++-12-dev
sudo apt-get update && sudo apt-get install cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build zlib1g-dev clang-12
```     

## How to run

Evaluation

```
cd python/mujoco_mpc/exp
python eval_gui.py
```

Batch Evaluation

```
python batch_evaluation.py
```

## How to deploy

### Sim2Sim

```
cd python/mujoco_mpc/deploy
python sim.py

cd python/mujoco_mpc/deploy
python control.py
```

### Sim2Real

Real robot ip and mask:
```
192.168.123.222
255.255.255.0
```
verify by `ping 192.168.123.161`

```
cd python/mujoco_mpc/deploy
python mocap.py
python real.py

cd python/mujoco_mpc/deploy
python control.py
```

## How to add new robot

1. match the site name and position
2. match the actuator, please use position control and use damping in joint rather than kv. 

## Notes

1. for pd position controller, try to use damping instead of kv to make the system more stable.
2. The output of the MPC controller for g1 robot is scaled control is scaled torque, so remember to scale up the control. 


