cube_slam在orb_slam的基础上引入物体级别的特征，从而给slam带来了更多可能。   
cube_slam具备以下亮点
- 添加物体级别特征
- 添加运动约束对动态物体建模
- 通过地面恢复单目尺度
- 除了点相机约束外，添加点物体约束、点物体相机约束等

## 本项目添加的功能
- 静态场景
    - 在线3d目标检测
- 动态场景
    - 在线3d目标检测
    - bytetrack的目标跟踪
    - 针对kitti数据集，对物体添加高度和运动方向一致性约束(物体-物体约束)

## 待解决的问题
1. 需要进一步优化动态场景
2. 由于需要预先知道物体的真实尺寸(长宽高)，本代码中只考虑car，可以根据需求修改

## b战视频demo
- [语义cube_slam之静态场景](https://www.bilibili.com/video/BV1Vt421W7GN)



## how to start 
```
# export yolo's onnx
pip install ultralytics
# https://docs.ultralytics.com/modes/export/#introduction
# https://github.com/ultralytics/assets/releases/ 

yolo export model=yolov8m.pt format=onnx imgsz=640,640 opset=15
yolo export model=yolov8m-seg.pt format=onnx imgsz=640,640 opset=15

# c++ dependency
- opencv-4.8 and opencv_contrib-4.8
- pcl-1.13
- eigen-3.3.7

git clone https://github.com/lturing/cube_slam_modified
cd cube_slam_modified
mkdir build
cd build 
cmake .. && make -j6 && cd ..

# 静态场景
# kitti_00
./Examples/bin/mono_kitti ./Vocabulary/ORBvoc.txt Examples/config/KITTI00-02_me.yaml /home/spurs/dataset/kitti_rgb_00

# 动态场景
## https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip

./Examples/bin/mono_kitti ./Vocabulary/ORBvoc.txt Examples/config/KITTI_10_03.yaml /home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02

# 在线3d框检测
./Examples/bin/detect_2d_lines_3d_cuboid

# yolov8 检测和分割
./Examples/bin/yolov8_detect_segment

# yolov8 检测和跟踪
## 基于detect
/Examples/bin/yolov8_detect_track
## 基于segmentation
./Examples/bin/yolov8_segment_track

```


## ref
- [cube_slam](https://github.com/shichaoy/cube_slam.git)
- [cube_slam_yolov5](/https://github.com/Geralt-of-winterfall/cube_slam)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)

## future work
- [point-plane-object-SLAM](https://github.com/benchun123/point-plane-object-SLAM)
- [EAO-SLAM](https://github.com/yanmin-wu/EAO-SLAM.git)


<br>
<details>
  <summary><strong>offical readme</strong>(click to expand)</summary>

# Cube SLAM #
This code contains two mode:
1)  object SLAM integrated with ORB SLAM. See ```orb_object_slam```  Online SLAM with ros bag input. It reads the offline detected 3D object.
2) Basic implementation for Cube only SLAM. See ```object_slam``` Given RGB and 2D object detection, the algorithm detects 3D cuboids from each frame then formulate an object SLAM to optimize both camera pose and cuboid poses.  is main package. ```detect_3d_cuboid``` is the C++ version of single image cuboid detection, corresponding to a [matlab version](https://github.com/shichaoy/matlab_cuboid_detect).

**Authors:** [Shichao Yang](https://shichaoy.github.io./)

**Related Paper:**

* **CubeSLAM: Monocular 3D Object SLAM**, IEEE Transactions on Robotics 2019, S. Yang, S. Scherer  [**PDF**](https://arxiv.org/abs/1806.00557)

If you use the code in your research work, please cite the above paper. Feel free to contact the authors if you have any further questions.



## Installation

### Prerequisites
This code contains several ros packages. We test it in **ROS indigo/kinetic, Ubuntu 14.04/16.04, Opencv 2/3**. Create or use existing a ros workspace.
```bash
mkdir -p ~/cubeslam_ws/src
cd ~/cubeslam_ws/src
catkin_init_workspace
git clone git@github.com:shichaoy/cube_slam.git
cd cube_slam
```

### Compile dependency g2o
```bash
sh install_dependenices.sh
```


### Compile
```bash
cd ~/cubeslam_ws
catkin_make -j4
```


## Running #
```bash
source devel/setup.bash
roslaunch object_slam object_slam_example.launch
```
You will see results in Rviz. Default rviz file is for ros indigo. A kinetic version is also provided.

To run orb-object SLAM in folder ```orb_object_slam```, download [data](https://drive.google.com/open?id=1FrBdmYxrrM6XeBe_vIXCuBTfZeCMgApL). See correct path in ```mono.launch```, then run following in two terminal:
``` bash
roslaunch orb_object_slam mono.launch
rosbag play mono.bag --clock -r 0.5
```

To run dynamic orb-object SLAM mentioned in the paper, download [data](https://drive.google.com/drive/folders/1T2PmK3Xt5Bq9Z7UhV8FythvramqhOo0a?usp=sharing). Similar to above, set correct path in ```mono_dynamic.launch```, then run the launch file with bag file.


If compiling problems met, please refer to ORB_SLAM.


### Notes

1. For the online orb object SLAM, we simply read the offline detected 3D object txt in each image. Many other deep learning based 3D detection can also be used similarly especially in KITTI data.

2. In the launch file (```object_slam_example.launch```), if ```online_detect_mode=false```, it requires the matlab saved cuboid images, cuboid pose txts and camera pose txts.  if ```true```, it reads the 2D object bounding box txt then online detects 3D cuboids poses using C++.

3. ```object_slam/data/``` contains all the preprocessing data. ```depth_imgs/``` is just for visualization. ```pred_3d_obj_overview/``` is the offline matlab cuboid detection images. ```detect_cuboids_saved.txt``` is the offline cuboid poses in local ground frame, in the format "3D position, 1D yaw, 3D scale, score". ```pop_cam_poses_saved.txt``` is the camera poses to generate offline cuboids (camera x/y/yaw = 0, truth camera roll/pitch/height) ```truth_cam_poses.txt``` is mainly used for visulization and comparison.

	```filter_2d_obj_txts/``` is the 2D object bounding box txt. We use Yolo to detect 2D objects. Other similar methods can also be used. ```preprocessing/2D_object_detect``` is our prediction code to save images and txts. Sometimes there might be overlapping box of the same object instance. We need to filter and clean some detections. See the [```filter_match_2d_boxes.m```](https://github.com/shichaoy/matlab_cuboid_detect/blob/master/filter_match_2d_boxes.m) in our matlab detection package.

</details>