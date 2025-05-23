# Revisiting Neighbourhoods in Mean Field Reinforcement Learning

Code base for the NeurIPS 2025 submission (Submission Number 18718): Revisiting Neighbourhoods in Mean Field Reinforcement Learning

 
## Code structure


- See folder Algorithms for all algorithmic implementations. 

- See folder MAgent for MAgent environment experiments.

- See folder Neural MMO for Neural MMO environment experiments.

- See folder SMARTS for SMARTS environment experiments.

- See folder COVID19task for experiments on the COVID19 domain.




## Installation Instructions for Ubuntu 18.04



### MAgent Environments 

##### Requirements

Atleast 

- `python==3.8`
- `gym==0.20.0`
- `matplotlib`


#### Install MARLlib 

You can get more help from [MARLlib](https://marllib.readthedocs.io/en/latest/handbook/installation.html)

```shell
conda create -n marllib python=3.8
conda activate marllib
git clone https://github.com/Replicable-MARL/MARLlib.git
cd MARLlib
pip install --upgrade pip
pip install -r requirements.txt

# we recommend the gym version between 0.20.0~0.22.0.
pip install gym>=0.20.0,<0.22.0

# add patch files to MARLlib
python patch/add_patch.py -y
```

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

2. Run training script for training (e.g. IL):

    ```shell
    python3 battleIL.py
    ```

    or get help:

    ```shell
    python3 battleIL.py --help
    ```


Similarly you can run the experiments for the Combined Arms and Tiger environments. 






### Neural MMO Environments


## Install
```bash
conda create -n nmmo python==3.9
conda activate nmmo
cd ./NeuralMMOimplementation
apt install git-lfs
cd ./NeuralMMOenvironment
make install
cd ./NeuralMMOimplementation
pip install -r requirements_tool.txt
```


Use the appropriate algorithm from the Algorithms folder in submission/submission.py and then you are ready for training. 

## Train and Evaluate
```bash
cd ./NeuralMMOimplementation
python tool.py
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

# Setup virtual environment. Presently at least Python 3.8 and higher is officially supported.
python3.8 -m venv .venv

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

The code for training and testing uses the `examples/multi_agent.py` file. Different scenarios from the scenarios folder (based on sumo) is used for training. Different agents are build from each MFA and baseline algorithms in the multi\_agent.py file. 

Use the appropriate algorithm from the Algorithms folder in the multi\_agent.py file and then you are ready for training.

```bash 
cd <path>/SMARTS
scl run --envision examples/multi_agent.py scenarios/sumo/loop
```




The `--envision` flag runs the Envision server which displays the simulation visualization. 


After executing the above command, visit http://localhost:8081/ to view the experiment.


```bash
scl run --envision <examples/path> <scenarios/path> 
```




### COVID-19 Vaccination task environments

Please download the required data (~10GB) from this [link](https://drive.google.com/drive/folders/1-68jPOd6NXVyiC1PWbo-9wrqiktOi4GT?usp=sharing) and substitute this in the data folder. All other libraries required for this experiment should already be installed when installing the libraries for the previous environments.  

The simulator for this experiment can be found in `COVID19task/code/simulator`. 

For training and evaluation use the following command: 

```bash
python code/train.py
```

To run various algorithms, replace the MARL class in the train.py file with the appropriate algorithm from the Algorithms folder. 




## Code Citations

We would like to cite [MAgent](https://github.com/geek-ai/MAgent) for code providing the environments used in the MAgent experiments. 

We would also like to cite [SMARTS](https://github.com/huawei-noah/SMARTS) for code providing the environments used in the SMARTS experiments. 

We would also like to cite [Neural MMO](https://github.com/neuralmmo) for code providing the environments used in the Neural MMO experiments. 

We would also like to cite [MARLlib](https://marllib.readthedocs.io/en/latest/) for implementations of several baselines used in the paper. 

We would also like to cite [GAT-MF](https://github.com/tsinghua-fib-lab/Large-Scale-MARL-GATMF) for implementations of GAT-MF and the associated COVID-19 vaccination task.  

All of the above mentioned repositories use the MIT license, except the GAT-MF codebase which does not include a license. 


