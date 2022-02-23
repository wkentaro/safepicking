<h1 align="center">SafePicking</h1>
<h3 align="center">Learning Safe Object Extraction via Object-Level Mapping</h3>

<p align="center">
  <a href="https://wkentaro.com">Kentaro Wada</a>,
  <a href="https://stepjam.github.io">Stephen James</a>,
  <a href="https://www.doc.ic.ac.uk/~ajd/">Andrew J. Davison</a>
  <br/>
  <a href="https://www.imperial.ac.uk/dyson-robotics-lab/">Dyson Robotics Laboratory</a>,
  <a href="https://www.imperial.ac.uk/">Imperial College London</a>
  <br/>
  IEEE International Conference on Robotics and Automation (ICRA), 2022
</p>

<p align="center">
  <a href="#installation"><b>Installation</b></a> |
  <a href="#usage"><b>Usage</b></a> |
  <a href="https://arxiv.org/abs/2202.05832"><b>Paper</b></a> |
  <a href="https://youtu.be/ejjqiBqRRKo"><b>Video</b></a> |
  <a href="https://safepicking.wkentaro.com"><b>Webpage</b></a>
</p>

<div align="center">
  <img src="docs/assets/img/teaser_horizontal.png" width="50%">
</div>

---

<div align="center">
  <img src="docs/assets/img/whole_pipeline.gif" width="32%">
  <img src="docs/assets/img/vs_heuristic.gif" width="32%">
  <img src="docs/assets/img/real_ablation.gif" width="32%">
</div>

## Installation

### Python project only

```bash
make install

source .anaconda3/bin/activate
./checks/check_motion_planning.py
```

### ROS project

- `robot-agent`: A computer with CUDA and a GPU installed for visual processing.
- `robot-node`: A computer with a real-time OS for a Panda robot.

#### @robot-agent

```bash
make install
source .anaconda3/bin/activate

cd ros/
make install
source devel/setup.sh
```

#### @robot-node

```
cd ros/
source /opt/ros/noetic/setup.sh

catkin build morefusion_panda
rosrun morefusion_panda create_udev_rules.sh

catkin build safepicking_ros
```

## Usage

### Training & Inference

```bash
cd examples/picking/

# download pile files at ~/.cache/safepicking/pile_generation/
./download_piles.py

./train.py --model fusion_net --noise
./learned.py logs/XXX/weights/YYY

# use pretrained model
./download_pretrained_models.py

# inference in the test environments: ~/.cache/safepicking/pile_generation/00009000 - 00009999.pkl
./learned.py --weight-dir logs/20210709_005731-fusion_net-noise/weights/84500 \
             ~/.cache/safepicking/pile_generation/00009000.pkl
```

<div>
  <img src="docs/assets/img/learned_00009001.gif" width="18%">
  <img src="docs/assets/img/learned_00009005.gif" width="18%">
  <img src="docs/assets/img/learned_00009007.gif" width="18%">
  <img src="docs/assets/img/learned_00009010.gif" width="18%">
  <img src="docs/assets/img/learned_00009013.gif" width="18%">
</div>

### Robotic demonstration

```bash
robot-node  $ roslaunch safepicking_ros panda_control.launch

robot-agent $ roslaunch safepicking_ros setup.launch
robot-agent $ rosrun safepicking_ros safepicking_task_interface.py
>>> self.run([YcbObject.PITCHER])
```

<img src="docs/assets/img/robotic_demo.gif" width="90%">

## Citation

```
@inproceedings{Wada:etal:ICRA2022a,
  title={{SafePicking}: Learning Safe Object Extraction via Object-Level Mapping},
  author={Kentaro Wada and Stephen James and Andrew J. Davison},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2022},
}
```
