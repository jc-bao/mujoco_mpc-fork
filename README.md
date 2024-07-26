
Env setup: for Ubuntu 20.04

Build

```bash
sudo apt-get update && sudo apt-get install cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build zlib1g-dev clang-12
```
```bash
mkdir build
cd build
```

For Ubuntu 22.04, according to [this issue](sudo apt-get install libstdc++-12-dev), should fix for clang 12:

```bash
sudo apt-get install libstdc++-12-dev
```

```bash
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DCMAKE_C_COMPILER:STRING=clang-12 -DCMAKE_CXX_COMPILER:STRING=clang++-12 -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON
```
**Note: gRPC is a large dependency and can take 10-20 minutes to initially download.**

```bash
cmake --build . --config=Release
```
Run GUI application

```bash
cd bin
./mjpc
```

**Python API**

1. Build MJPC (see instructions above).
2. Python 3.10
3. (Optionally) Create a conda environment with **Python 3.10**:
```sh
conda create -n mjpc python=3.10
conda activate mjpc
```
4. Install MuJoCo
```sh
pip install mujoco
```
### Install API
Next, change to the python directory:
```sh
cd python
```
Install the Python module:
```sh
python setup.py install
```
Test that installation was successful:
```sh
python "mujoco_mpc/agent_test.py"
```

Example scripts are found in `python/mujoco_mpc/demos`. For example from `python/`:
```sh
python mujoco_mpc/demos/agent/cartpole_gui.py
```
will run the MJPC GUI application using MuJoCo's passive viewer via Python.