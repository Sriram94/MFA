# Revisiting Neighbourhoods in Mean Field Reinforcement Learning

Code base for the ICML 2025 submission (Submission Number 14387): Revisiting Neighbourhoods in Mean Field Reinforcement Learning

Note: This is a restricted version due to file size, licensing, and anonymity considerations. Full data and code will be
open-sourced with the paper.
 
## Code structure


- See folder MAgent for MAgent environment experiments.

- See folder Neural MMO for Neural MMO environment experiments.

- See folder SMARTS for SMARTS environment experiments.




## Installation Instructions for Ubuntu 18.04



### MAgent Environments 





##### Requirements

Atleast 

- `python==3.7.11`
- `gym==0.9.2`
- `matplotlib`


#### Compile MAgent platform and run

Before running Battle Game environment, you need to compile it. You can get more helps from: [MAgent](https://github.com/geek-ai/MAgent)

**Steps for compiling**

```shell
cd examples/battle_model
./build.sh
```

**Steps for training models under Battle Game settings**

1. Add python path in your `~/.bashrc` or `~/.zshrc`:

    ```shell
    vim ~/.zshrc
    export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
    source ~/.zshrc
    ```

2. Run training script for training (e.g. mfac):

    ```shell
    python3 train_battle.py --algo mfac
    ```

    or get help:

    ```shell
    python3 train_battle.py --help
    ```## Compile MAgent platform and run

Before running Battle Game environment, you need to compile it. You can get more helps from: [MAgent](https://github.com/geek-ai/MAgent)

**Steps for compiling**

```shell
cd examples/battle_model
./build.sh
```

**Steps for training models under Battle Game settings**

1. Add python path in your `~/.bashrc` or `~/.zshrc`:

    ```shell
    vim ~/.zshrc
    export PYTHONPATH=./examples/battle_model/python:${PYTHONPATH}
    source ~/.zshrc
    ```

2. Run training script for training (e.g. mfac):

    ```shell
    python3 train_battle.py --algo mfac
    ```

    or get help:

    ```shell
    python3 train_battle.py --help
    ```






### Neural MMO Environments


## Install
```bash
pip install git+http://gitlab.aicrowd.com/henryz/ijcai2022nmmo.git
pip install -r requirements.txt
```


## Train and evaluation
```bash
cd monobeast/training

# train
bash train.sh

# plot
python plot.py

# local evaluation
cd monobeast/my-submission
python eval.py
```



### SMARTS Environments

#### Documentation
Documentation is available at [smarts.readthedocs.io](https://smarts.readthedocs.io/en/latest).

#### Setup


```bash
git clone https://github.com/huawei-noah/SMARTS.git
cd <path/to/SMARTS>

# For Mac OS X users, ensure XQuartz is pre-installed.
# Install the system requirements. You may use the `-y` option to enable automatic assumption of "yes" to all prompts to avoid timeout from waiting for user input. 
bash utils/setup/install_deps.sh

# Setup virtual environment. Presently at least Python 3.7 and higher is officially supported.
python3.7 -m venv .venv

# Enter virtual environment to install dependencies.
source .venv/bin/activate

# Upgrade pip.
pip install --upgrade pip

# Install smarts with extras as needed. Extras include the following: 
# `camera-obs` - needed for rendering camera sensor observations, and for testing.
# `test` - needed for testing.
# `train` - needed for RL training and testing.
pip install -e '.[camera-obs,test,train]'

# Run sanity-test and verify they are passing.
# If tests fail, check './sanity_test_result.xml' for test report. 
make sanity-test
```

Use the `scl` command to run SMARTS together with it's supporting processes. 

To run the default example, firstly build the scenario `scenarios/sumo/loop`.
```bash
scl scenario build --clean scenarios/sumo/loop
```

Then, run a single-agent SMARTS simulation with Envision display and `loop` scenario.
```bash 
scl run --envision examples/single_agent.py scenarios/sumo/loop 
```

The `--envision` flag runs the Envision server which displays the simulation visualization. See [./envision/README.md](./envision/README.md) for more information on Envision, SMARTS's front-end visualization tool.

After executing the above command, visit http://localhost:8081/ to view the experiment.


```bash
scl run --envision <examples/path> <scenarios/path> 
```








Now you can just run the respective files mentioned in the above section to run our code.
