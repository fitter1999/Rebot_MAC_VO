# <div align="center">MAC-VO: Metrics-aware Covariance for Learning-based Stereo Visual Odometry</div>

### <div align="center">🥇 ICRA 2025 Best Conference Paper Award<br/>🥇 ICRA 2025 Best Paper Award on Robot Perception</div>

<p align="center">
  <a href="https://mac-vo.github.io"><img src="https://img.shields.io/badge/Homepage-4385f4?style=flat&logo=googlehome&logoColor=white"></a>
  <a href="https://arxiv.org/abs/2409.09479v2"><img src="https://img.shields.io/badge/arXiv-b31b1b?style=flat&logo=arxiv&logoColor=white"></a>
  <a href="https://www.youtube.com/watch?v=O_HowJk-GDw"><img src="https://img.shields.io/badge/YouTube-b31b1b?style=flat&logo=youtube&logoColor=white"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>


<p align="center">
  <img src="asset/ICRAvideo.gif" alt="ICRA floor 3" width="600" />
</p>


> [!NOTE]  
> We plan to release TensorRT accelerated implementation and adapting more matching networks for MAC-VO. If you are interested, please star ⭐ this repo to stay tuned.

> [!NOTE]
>
> We provide **[documentation for extending MAC-VO](https://mac-vo.github.io/wiki/)** for extending MAC-VO or using this repository as a boilerplate for *your* learning-based Visual Odometry.
>

## Rebot MAC-VO 实时双目使用教程

本仓库在官方 MAC-VO 基础上增加了面向本机实时双目相机的脚本，重点支持：

- DECXIN-3261V 单 USB 拼接双目相机
- OV2710 双 USB 双目相机
- Miniforge/Micromamba 本地环境运行
- Rerun 实时显示相机轨迹、图像、点云地图
- `Ctrl+C` 后保存轨迹、VO 点云和 mapping 点云
- 保存结果重新导入 Rerun，按帧重放地图增长

### 1. 环境和模型

本机推荐使用 `run_macvo_wjy.sh` 封装的 `macvo_wjy` 环境：

```bash
cd /home/wjy/WJY/MAC-VO
./run_macvo_wjy.sh python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

模型文件需要放在 `Model/` 下：

```bash
mkdir -p Model
wget -O Model/MACVO_FrontendCov.pth https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth
wget -O Model/MACVO_posenet.pkl https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_posenet.pkl
```

`Model/` 不提交到 Git，克隆后需要重新下载。

### 2. DECXIN-3261V 相机检查

DECXIN-3261V 是一个 UVC 设备，单帧 `4000x1200`，左右目横向拼接：

- 左目裁剪：`x=2080..3999`
- 右目裁剪：`x=160..2079`
- 基线：`60 mm`
- 默认 VO 输入：`640x480`
- 默认采集后端：`gst-v4l2`

检查设备：

```bash
lsusb
ls -l /dev/video* /dev/v4l/by-id /dev/v4l/by-path
gst-device-monitor-1.0 Video/Source
```

正常应看到：

```text
1bcf:2d50 Sunplus Innovation Technology Inc. DECXIN Camera
image/jpeg, width=4000, height=1200, framerate={60/1,30/1}
```

纯采集测试：

```bash
./run_decxin3261v_live_wjy.sh \
  --capture-only \
  --max-frames 120 \
  --status-every 2
```

如果只测试相机 60 FPS 能力：

```bash
./run_decxin3261v_live_wjy.sh \
  --camera-fps 60 \
  --capture-only \
  --max-frames 120 \
  --status-every 2
```

注意：实时 VO 建议使用 30 FPS 采集。`4000x1200@60` 会增加 MJPG 解码和内存带宽压力，可能和 MAC-VO 前端推理抢资源。

### 3. 屏幕棋盘格标定

如果没有打印棋盘格，可以用仓库里的 HTML 棋盘格：

```bash
xdg-open Tools/calibration_board.html
```

DECXIN 默认使用 `28 mm` 方格。采集标定图：

```bash
Scripts/AdHoc/DECXIN3261V/capture_calibration.sh
```

执行标定：

```bash
Scripts/AdHoc/DECXIN3261V/calibrate.sh
```

生成结果：

```text
Calibration/decxin3261v_screen_640x480/calibration_result/calibration.yaml
Calibration/decxin3261v_screen_640x480/calibration_result/rectify_maps.npz
```

OV2710 的流程在 `Scripts/AdHoc/OV2710/` 和对应顶层脚本中，默认实时入口是：

```bash
./run_ov2710_live_wjy.sh --useRR
```

### 4. DECXIN 实时 VO 和建图

只跑实时 VO：

```bash
./run_decxin3261v_live_wjy.sh --useRR
```

实时 mapping，Rerun 显示图像、相机轨迹和逐帧地图点云：

```bash
./run_decxin3261v_mapping_wjy.sh --useRR
```

同时保存逐帧输入图像，便于后处理和 Rerun 回放显示 image：

```bash
./run_decxin3261v_mapping_wjy.sh \
  --useRR \
  --record-sequence
```

运行结束时按一次 `Ctrl+C`，等待终端出现：

```text
Saved trajectory and map to Results_decxin3261v_mapping/...
```

保存目录里主要文件：

```text
poses.npy
tensor_map.npz
preview/
stereo_sequence/     # 只有加 --record-sequence 才有
```

### 5. 重新打开保存地图到 Rerun

已有结果不需要接相机，直接用保存目录里的 `poses.npy`、`tensor_map.npz` 和可选的 `stereo_sequence/` 重建 Rerun 视图。

打开最新一次保存结果：

```bash
./run_decxin3261v_view_map_wjy.sh
```

打开最新一次结果，并按帧重放地图增长、黄色轨迹、相机位置和 image：

```bash
./run_decxin3261v_view_map_wjy.sh --growth
```

指定某次已有结果逐帧打开：

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/<time_dir> \
  --growth
```

打开离线建图结果目录，脚本会自动查找里面最新的 `tensor_map.npz`：

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_offline_mapping \
  --growth
```

如果离线结果的 `config.yaml` 里记录了原始 `stereo_sequence` 路径，脚本会自动读取原始逐帧图像。否则只能显示点云、黄色轨迹和相机位姿。

例如：

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/07_14_015422 \
  --growth
```

如果点云太多导致 Rerun 卡顿，可以降低显示点数或跳帧：

```bash
./run_decxin3261v_view_map_wjy.sh \
  --result Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/<time_dir> \
  --growth \
  --every 2 \
  --image-every 2 \
  --max-points 150000
```

如果没有 `stereo_sequence/`，脚本只能显示 `preview/first_pair_left.png`，不能逐帧显示 image。需要在建图时加 `--record-sequence`。

### 6. 导出 PLY 点云

导出重建点云：

```bash
Scripts/AdHoc/DECXIN3261V/export_map.sh \
  --result Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/<time_dir> \
  --max-distance 6.0 \
  --cov-det-percentile 95
```

输出目录：

```text
reconstruction/mapping_points_all.ply
reconstruction/mapping_points_filtered.ply
reconstruction/trajectory.ply
reconstruction/topdown_preview.png
```

更干净的点云：

```bash
Scripts/AdHoc/DECXIN3261V/export_map.sh \
  --result Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/<time_dir> \
  --output-dir Results_decxin3261v_mapping/MACVO-DECXIN3261V-Mapping@DECXIN3261V-live/<time_dir>/reconstruction_clean \
  --max-distance 4.5 \
  --cov-det-percentile 90
```

### 7. 电池模式性能

拔掉电源后，笔记本通常会进入 CPU/GPU/USB 省电策略，MAC-VO 前端推理可能从几百毫秒变成数秒。运行前可以切换到移动性能模式：

```bash
Scripts/AdHoc/DECXIN3261V/set_mobile_performance.sh
```

运行结束恢复省电：

```bash
Scripts/AdHoc/DECXIN3261V/set_mobile_powersave.sh
```

如果电池模式仍然卡顿，降低实时负载：

```bash
./run_decxin3261v_mapping_wjy.sh \
  --useRR \
  --record-sequence \
  --vo-fps 2 \
  --rr-every 2 \
  --rr-max-points 5000
```

### 8. 常用排错

相机被会议软件占用：

```bash
pgrep -af "wemeet|Tencent|obs|rerun|Run_Realtime"
```

Rerun 残留：

```bash
pgrep -af rerun
kill <pid>
```

确认采集帧率：

```bash
./run_decxin3261v_live_wjy.sh --capture-only --max-frames 120 --status-every 2
```

确认 GPU 状态：

```bash
nvidia-smi
```

## 🔥 Updates

* [Nov 2025] We release the trajectories we collected with ZedX Stereo camera on ICRA 2025 conference. See the *Additional Trajectory Release* in README for more details.
* [Jun 2025] We release the **MAC-VO Fast Mode** - with faster pose graph optimization and mixed-precision inference, we achieve 2x speedup compared to previous version and reach speed of 12.5fps on 480x640 images. 

  See `Config/Experiment/MACVO/MACVO_Fast.yaml` for detail. 
  
  Original example is also boosted from 5fps to 7fps and the config file is moved to `MACVO_Performant.yaml`.
* [Apr 2025] Our work was nominated as the **ICRA 2025 Best Paper Award Finalist** (top 1%)! Keep an eye on our presentation on May 20, 16:35-16:40 Room 302. We also plan to provide a real-world demo at the conference.
* [Mar 2025] We boost the performance of MAC-VO with a new backend optimizer, the MAC-VO now also supports *dense mapping* without any additional computation.
* [Jan 2025] Our work is accepted by the IEEE International Conference on Robotics and Automation (ICRA) 2025. We will present our work at ICRA 2025 in Atlanta, Georgia, USA.
* [Nov 2024] We released the ROS-2 integration at https://github.com/MAC-VO/MAC-VO-ROS2 along with the documentation at https://mac-vo.github.io/wiki/ROS/

## Download the Repo

Clone the repository using the following command to include all submodules automatically.

`git clone https://github.com/MAC-VO/MAC-VO.git --recursive`


## 🔧 Minimum Requirements

| Component        | Minimum Version | Notes                                            |
|------------------|-----------------|--------------------------------------------------|
| **CUDA Runtime** | ≥ 12.4          | Dockerfile installs correct version              |
| **Python**       | ≥ 3.10          |                                                  |
| **VRAM**         | ≥ 6 GB          | 640×480; fast mode (mixed precision) needs 2.7GB |


## 📦 Installation & Environment

### Environment

1. **Docker Image**

    ```bash
    $ docker build --network=host -t macvo:latest -f Docker/Dockerfile .
    ```

2. **Virtual Environment**

    You can setup the dependencies in your native system. MAC-VO codebase can only run on Python 3.10+. See `requirements.txt` for environment requirements.

    <details>
      <summary>How to adapt MAC-VO codebase to Python &lt; 3.10?</summary>
      
      The Python version requirement we required is mostly due to the [`match`](https://peps.python.org/pep-0634/) syntax used and the [type annotations](https://peps.python.org/pep-0604/).

      The `match` syntax can be easily replaced with `if ... elif ... else` while the type annotations can be simply removed as it does not interfere runtime behavior.
    </details>

### Pretrained Models

All pretrained models for MAC-VO, stereo TartanVO and DPVO are in our [release page](https://github.com/MAC-VO/MAC-VO/releases/tag/model). Please create a new folder `Model` in the root directory and put the pretrained models in the folder.

    $ mkdir Model
    $ wget -O Model/MACVO_FrontendCov.pth https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth
    $ wget -O Model/MACVO_posenet.pkl https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_posenet.pkl

## 🚀 Quick Start: Run MAC-VO on Demo Sequence

Test MAC-VO immediately using the provided demo sequence. The demo sequence is a selected from the TartanAir v2 dataset.

### 1/4 Download the Data

1. Download a demo sequence through [Google Drive](https://drive.google.com/file/d/1kCTNMW2EnV42eH8g2STJHcVWEbVKbh_r/view?usp=sharing).
2. Download pre-trained model for [frontend model](https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_FrontendCov.pth) and [posenet](https://github.com/MAC-VO/MAC-VO/releases/download/model/MACVO_posenet.pkl).

### 2/4 Start the Docker
To run the Docker: 

    $ docker run --gpus all -it --rm  -v [DATA_PATH]:/data -v [CODE_PATH]:/home/macvo/workspace macvo:latest

To run the Docker with visualization: 

    $ xhost +local:docker; docker run --gpus all -it --rm  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix  -v [DATA_PATH]:/data -v [CODE_PATH]:/home/macvo/workspace macvo:latest


### 3/4 Run MAC-VO

We will use `Config/Experiment/MACVO/MACVO_example.yaml` as the configuration file for MAC-VO.

1. Change the `root` in the data config file 'Config/Sequence/TartanAir_example.yaml' to reflect the actual path to the demo sequence downloaded.
2. Run with one of the following command:

    *Performant Mode* - best performance with moderate speed (7.5fps on 480x640 image)

    ```bash
    $ cd workspace
    $ python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_Performant.yaml --data Config/Sequence/TartanAir_example.yaml
    ```

    *Fast Mode* - slightly degraded performance (<5% increase in RTE and ROE) with most speed (12.5fps on 480x640 image)

    ```bash
    $ cd workspace
    $ python3 MACVO.py --odom Config/Experiment/MACVO/MACVO_Fast.yaml --data Config/Sequence/TartanAir_example.yaml
    ```

> [!NOTE]
>
> See `python MACVO.py --help` for more flags and configurations.
>
> The demo sequence is RGB‑only. If your dataset includes depth.npy and/or flow.npy, set both flags to true.
>

### 4/4 Visualize and Evaluate Result

Every run will produce a `Sandbox` (or `Space`). A `Sandbox` is a storage unit that contains all the results and meta-information of an experiment. The evaluation and plotting script usually requires one or more paths of sandbox(es).

#### **Evaluate Trajectory**

  Calculate the absolute translate error (ATE, m); relative translation error (RTE, m/frame); relative orientation error (ROE, deg/frame); relative pose error (per frame on se(3)).

  ```bash
  $ python -m Evaluation.EvalSeq --spaces SPACE_0, [SPACE, ...]
  ```

#### **Plot Trajectory**

  Plot sequences, translation, translation error, rotation and rotation error.

  ```bash
  $ python -m Evaluation.PlotSeq --spaces SPACE_0, [SPACE, ...]
  ```

## 🛠️ Additional Commands and Utility

* **Run MAC-VO (*Ours* method) on a Single Sequence**
    ```bash
    $ python MACVO.py --odom ./Config/Experiment/MACVO/MACVO.yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
    ```

* **Run MAC-VO for Ablation Studies**
    ```bash
    $ python MACVO.py --odom ./Config/Experiment/MACVO/Ablation_Study/[CHOOSE_ONE_CFG].yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
    ```

* **Run MAC-VO on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_MACVO --odom [PATH_TO_ODOM_CONFIG]
  ```

* **Run MAC-VO Mapping Mode**

  ```bash
  $ python MACVO.py --odom ./Config/Experiment/MACVO/MACVO_MappingMode.yaml --data ./Config/Sequence/TartanAir_abandonfac_001.yaml
  ```

### 📊 Plotting and Visualization

We used [the Rerun](https://rerun.io) visualizer to visualize 3D space including camera pose, point cloud and trajectory.

* **On Machine with GUI**

  1. Run `MACVO.py` with the following command line
    
        ```bash
        $ python MACVO.py --useRR --odom [ODOM_CONFIG] --data [DATA_CONFIG]
        ```
     
        A rerun visualizer should pop up with the trajectory and *per-frame* point cloud & tracking features visualized.
  2. To accumulate the point cloud for dense mapping visualization, please follow the instruction here: https://github.com/MAC-VO/MAC-VO/issues/4#issuecomment-2495620352

* **On Headless Machine**

  1. Install the `rerun_sdk` python package on both your machine (with GUI) and remote headless environment. Also setup a port forwarding from remote port `9877` to your local machine port `9877`.
  2. Start a rerun server by rerun --serve & on the headless machine
  3. On your machine (with GUI), run rerun ws://localhost:9877 to connect to the remote visualization server. You should see "2 sources connected" on the top right corner of visualizer if everything works smoothly.
  4. On the headless machine, run
      ```bash
      $ python MACVO.py --useRR --odom [ODOM_CONFIG] --data [DATA_CONFIG]
      ```
  5. To accumulate the point cloud for dense mapping visualization, please follow the instruction here: https://github.com/MAC-VO/MAC-VO/issues/4#issuecomment-2495620352

### 📈 Baseline Methods

We also integrated two baseline methods (DPVO, TartanVO Stereo) into the codebase for evaluation, visualization and comparison.

<details>
<summary>
Expand All (2 commands)
</summary>

* **Run DPVO on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_DPVO --odom ./Config/Experiment/Baseline/DPVO/DPVO.yaml
  ```

* **Run TartanVO (Stereo) on Test Dataset**

  ```bash
  $ python -m Scripts.Experiment.Experiment_TartanVO --odom ./Config/Experiment/Baseline/TartanVO/TartanVOStereo.yaml
  ```

</details>


## 🤗 Customization, Extension and Future Developement

> This codebase is designed with *modularization* in mind so it's easy to modify, replace, and re-configure modules of MAC-VO. One can easily use or replase the provided modules like flow estimator, depth estimator, keypoint selector, etc. to create a new visual odometry.

We welcome everyone to extend and redevelop the MAC-VO. For documentation please visit the [Documentation Site](https://mac-vo.github.io/wiki/)

### Custom Data Format

To test MAC-VO on your custom data format, you use `GeneralStereo` dataloader class in `DataLoader/Dataset/GeneralStereo.py` as a starting point.

This dataloader class corresponds to the `Config/Sequence/Example_GeneralStereo.yaml` configuration file, where you can manually set the camera intrinsic and stereo basline etc.

### Coordinate System in this Project

**PyTorch Tensor Data** - All images are stored in `BxCxHxW` format following the convention. Batch dimension is always the first dimension of tensor.

**Pixels on Camera Plane** - All pixel coordinates are stored in `uv` format following the OpenCV convention, where the direction of uv are "east-down". *Note that this requires us to access PyTorch tensor in `data[..., v, u]`* indexing.

**World Coordinate** - `NED` convention, `+x -> North`, `+y -> East`, `+z -> Down` with the first frame being world origin having identity SE3 pose.

## ➕ Additional Trajectories Release

Upon [request](https://github.com/MAC-VO/MAC-VO/issues/27) we released the Zed Stereo dataset we collected on the ICRA 2025 conference. You can now download them using the command:

```bash
pip install minio

python -m Scripts.AdHoc.Download_ICRA25_Zed_Data --dst [Download_Destination]
```

After download, unzip the trajectories and modify the path in `Config/Sequence/ICRA25_Zed_0250.yaml`. Run MAC-VO (fast mode) with:

```bash
python MACVO.py --data ./Config/Sequence/ICRA25_Zed_0250.yaml --odom ./Config/Experiment/MACVO/MACVO_Fast.yaml --useRR
```

> ⚠️ To reproduce the result we show on website and presentation, it is recommended to run the trajectory in its full resolution `980x980`.

## Citation / BibTex

If you find our work useful, please consider cite us with

```bibtex
@inproceedings{qiu2025mac,
 title={MAC-VO: Metrics-Aware Covariance for Learning-Based Stereo Visual Odometry mac-vo. github. io},
 author={Qiu, Yuheng and Chen, Yutian and Zhang, Zihao and Wang, Wenshan and Scherer, Sebastian},
 booktitle={2025 IEEE International Conference on Robotics and Automation (ICRA)},
 pages={3803--3814},
 year={2025},
 organization={IEEE}
}
```
