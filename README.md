# Feedback Sampling with Mujoco MPC

## How to run

Evaluation

```
cd python/mujoco_mpc/exp
python eval_gui.py
```

## Notes

1. for pd position controller, try to use damping instead of kv to make the system more stable.
2. The output of the MPC controller for g1 robot is scaled control 