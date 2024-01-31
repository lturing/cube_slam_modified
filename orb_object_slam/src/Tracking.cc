/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <mutex>
#include <vector>

#include "Frame.h"
#include "KeyFrame.h"
#include "Map.h"
#include "MapPoint.h"
#include "LocalMapping.h"
#include "Initializer.h"
#include "PnPsolver.h"
#include "LoopClosing.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Optimizer.h"
#include "Viewer.h"
#include "System.h"

#include "ORBmatcher.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "KeyFrameDatabase.h"

// by me
#include "detect_3d_cuboid/include/detect_3d_cuboid.h"
#include "detect_3d_cuboid/include/object_3d_util.h"
#include "Parameters.h"
#include "MapObject.h"
#include <math.h>
#include <ctime>
#include <algorithm>
#include <boost/filesystem.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

#include "STrack.h"
#include "data_struct.h"

using namespace std;

namespace ORB_SLAM2
{

// Define a new bool reader in order to accept "true/false"-like values.
bool Tracking::read_bool(const cv::FileNode &node, bool &value, bool default_value)
{
    std::string s(static_cast<std::string>(node));
    if(s == "y" || s == "Y" || s == "yes" || s == "Yes" || s == "YES" || s == "true" || s == "True" || s == "TRUE" || s == "on" || s == "On" || s == "ON")
    {
        value = true; 
        return true;
    }
    if(s == "n" || s == "N" || s == "no" || s == "No" || s == "NO" || s == "false" || s == "False" || s == "FALSE" || s == "off" || s == "Off" || s == "OFF")
    {
        value = false; 
        return true;
    }
    value = default_value;
    std::cout << "failed to parse value=" << s << std::endl;
    return false;
    //value= static_cast<int>(node);
}

Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase *pKFDB,
                   const string &strSettingPath, const int sensor) : mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
                                                                     mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys),
                                                                     mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    InitToGround = cv::Mat::eye(4, 4, CV_32F);
    // set initial camera pose wrt ground. by default camera parallel to ground, height=1.7 (kitti)
    double init_x, init_y, init_z, init_qx, init_qy, init_qz, init_qw;
    init_x = fSettings["init_x"];
    init_y = fSettings["init_y"];
    init_z = fSettings["init_z"];
    init_qx = fSettings["init_qx"];
    init_qy = fSettings["init_qy"];
    init_qz = fSettings["init_qz"];
    init_qw = fSettings["init_qw"];

    Eigen::Quaternionf pose_quat(init_qw, init_qx, init_qy, init_qz);
    Eigen::Matrix3f rot = pose_quat.toRotationMatrix(); //     The quaternion is required to be normalized
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
            InitToGround.at<float>(row, col) = rot(row, col);

    InitToGround.at<float>(0, 3) = init_x;
    InitToGround.at<float>(1, 3) = init_y;
    InitToGround.at<float>(2, 3) = init_z;
    nominal_ground_height = init_z;

    cv::Mat R = InitToGround.rowRange(0, 3).colRange(0, 3);
    cv::Mat t = InitToGround.rowRange(0, 3).col(3);
    cv::Mat Rinv = R.t();
    cv::Mat Ow = -Rinv * t;
    GroundToInit = cv::Mat::eye(4, 4, CV_32F);
    Rinv.copyTo(GroundToInit.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(GroundToInit.rowRange(0, 3).col(3));

    InitToGround_eigen = Converter::toMatrix4f(InitToGround);
    GroundToInit_eigen = Converter::toMatrix4f(GroundToInit);

    mpMap->InitToGround = InitToGround;
    mpMap->GroundToInit = GroundToInit.clone();
    mpMap->InitToGround_eigen = InitToGround_eigen;
    mpMap->InitToGround_eigen_d = InitToGround_eigen.cast<double>();
    mpMap->GroundToInit_eigen_d = GroundToInit_eigen.cast<double>();
    mpMap->GroundToInit_opti = GroundToInit.clone();
    mpMap->InitToGround_opti = InitToGround.clone();
    mpMap->RealGroundToMine_opti = cv::Mat::eye(4, 4, CV_32F);
    mpMap->MineGroundToReal_opti = cv::Mat::eye(4, 4, CV_32F);

    image_width = fSettings["Camera.width"];
    image_height = fSettings["Camera.height"];

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    Kalib.setIdentity();
    Kalib(0, 0) = fx;
    Kalib(0, 2) = cx;
    Kalib(1, 1) = fy;
    Kalib(1, 2) = cy;
    Kalib_f = Kalib.cast<float>();
    invKalib = Kalib.inverse();
    invKalib_f = Kalib_f.inverse();
    mpMap->Kalib = Kalib;
    mpMap->Kalib_f = Kalib_f;
    mpMap->invKalib_f = invKalib_f;

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps = 30;
    
    frame_rate = 1.0 / fps;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    cout << "- img width: " << image_width << std::endl;
    cout << "- img height: " << image_height << std::endl;
    if (DistCoef.rows == 5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;
    cout << "- GroundToInit: " << GroundToInit << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    read_bool(fSettings["mono_firstframe_truth_depth_init"], mono_firstframe_truth_depth_init, false);
    read_bool(fSettings["mono_allframe_obj_depth_init"], mono_allframe_Obj_depth_init, false);
    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::STEREO || mono_firstframe_truth_depth_init)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
        mThDepth = mbf * (float)fSettings["ThDepth"] / fx;
        cout << endl
             << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (mDepthMapFactor == 0)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }

    read_bool(fSettings["mono_firstframe_Obj_depth_init"], mono_firstframe_Obj_depth_init, false);
    read_bool(fSettings["bundle_object_opti"], bundle_object_opti, false);

    read_bool(fSettings["whether_detect_object"], whether_detect_object, false);
    read_bool(fSettings["whether_read_offline_cuboidtxt"], whether_read_offline_cuboidtxt, false);
    read_bool(fSettings["associate_point_with_object"], associate_point_with_object, false);

    read_bool(fSettings["ba_dyna_pt_obj_cam"], ba_dyna_pt_obj_cam, false);
    read_bool(fSettings["ba_dyna_obj_velo"], ba_dyna_obj_velo, false);
    read_bool(fSettings["ba_dyna_obj_cam"], ba_dyna_obj_cam, false);

    obj_det_2d_thre = fSettings["obj_det_2d_thre"];
    read_bool(fSettings["build_worldframe_on_ground"], build_worldframe_on_ground, false); // transform initial pose and map to ground frame
    read_bool(fSettings["triangulate_dynamic_pts"], triangulate_dynamic_pts, false);
    use_truth_trackid = false; // whether use ground truth tracklet ID.
    whether_save_online_detected_cuboids = false;
    whether_save_final_optimized_cuboids = false;

    if (whether_detect_object)
    {
        read_bool(fSettings["use_truth_trackid"], use_truth_trackid, false);
        if (!whether_read_offline_cuboidtxt)
        {
            detect_cuboid_obj = std::make_shared<detect_3d_cuboid>();
            detect_cuboid_obj->print_details = false; // false  true
            detect_cuboid_obj->set_calibration(Kalib);
            detect_cuboid_obj->whether_sample_bbox_height = false;
            detect_cuboid_obj->whether_sample_cam_roll_pitch = true;
        }

        if (!whether_read_offline_cuboidtxt)
        {
            read_bool(fSettings["whether_save_online_detected_cuboids"], whether_save_online_detected_cuboids, false);
            if (whether_save_online_detected_cuboids)
            {
                std::string save_object_pose_txt = base_data_folder + "/slam_output/orb_live_pred_objs_temp.txt";
                save_online_detected_cuboids.open(save_object_pose_txt.c_str());
            }
        }
        if (whether_read_offline_cuboidtxt)
        {
            cuboid_folder = fSettings["cuboid_folder"].string();
            ReadAllObjecttxt();
        }
            
    }

    if (whether_detect_object)
    {
        read_bool(fSettings["whether_save_final_optimized_cuboids"], whether_save_final_optimized_cuboids, false);
        if (whether_save_final_optimized_cuboids)
        {
            if (final_object_record_frame_ind == 1e5)
            {
                std::cout << "Please set final_object_record_frame_ind!!!" << std::endl;
                whether_save_final_optimized_cuboids = false;
            }
        }
    }
    bool load_truth_pose;
    read_bool(fSettings["load_truth_pose"], load_truth_pose, false);
    std::string truth_pose_txts = base_data_folder + "/pose_truth.txt";
    Eigen::MatrixXd truth_cam_poses(5, 8);
    if (load_truth_pose && read_all_number_txt(truth_pose_txts, truth_cam_poses))
    {
        mpMapDrawer->truth_poses.resize(truth_cam_poses.rows() / 10, 3);
        for (int i = 0; i < mpMapDrawer->truth_poses.rows(); i++)
        {
            mpMapDrawer->truth_poses.row(i) = truth_cam_poses.row(i * 10).segment(1, 3);
            if (build_worldframe_on_ground)
            {
                Eigen::Quaterniond pose_quat(truth_cam_poses(i * 10, 7), truth_cam_poses(i * 10, 4), truth_cam_poses(i * 10, 5), truth_cam_poses(i * 10, 6));
                Eigen::Matrix4d pose_to_init;
                pose_to_init.setIdentity();
                pose_to_init.block(0, 0, 3, 3) = pose_quat.toRotationMatrix();
                pose_to_init.col(3).head<3>() = Eigen::Vector3d(truth_cam_poses(i * 10, 1), truth_cam_poses(i * 10, 2), truth_cam_poses(i * 10, 3));
                Eigen::Matrix4d pose_to_ground = InitToGround_eigen.cast<double>() * pose_to_init;
                mpMapDrawer->truth_poses.row(i) = pose_to_ground.col(3).head<3>();
            }
        }
        cout << "Read sampled truth pose size for visualization:  " << mpMapDrawer->truth_poses.rows() << endl;
    }

    read_bool(fSettings["whether_dynamic_object"], whether_dynamic_object, false);
    read_bool(fSettings["remove_dynamic_features"], remove_dynamic_features, false);
    read_bool(fSettings["use_dynamic_klt_features"], use_dynamic_klt_features, false);
    read_bool(fSettings["enable_ground_height_scale"], enable_ground_height_scale, false);

    std::vector<std::string> _classNamesList = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    std::vector<std::string> dynamicObjectClass = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball"
    };

    auto batchSize = 1;
    auto inputSize = cv::Size(640, 640);
    if (whether_dynamic_object)
    {
        seg_folder = fSettings["seg_folder"].string();
        seg_onnx_path = fSettings["seg_onnx_path"].string();

        Frame::mySegOnnxRun = std::make_shared<Segmentor_ONNXRUNTIME>();
        Frame::mySegOnnxCV = std::make_shared<Segmentor_OpenCV_DNN>();

        Frame::mySegOnnxRun->LoadModel(seg_onnx_path);
        Frame::mySegOnnxRun->setClassNames(_classNamesList);
        Frame::mySegOnnxRun->setDynamicClassNames(dynamicObjectClass);
        Frame::mySegOnnxRun->setBatchSize(batchSize);
        Frame::mySegOnnxRun->setInputSize(inputSize);

        Frame::mySegOnnxCV->LoadModel(seg_onnx_path);
        Frame::mySegOnnxCV->setClassNames(_classNamesList);
        Frame::mySegOnnxCV->setDynamicClassNames(dynamicObjectClass);
        Frame::mySegOnnxCV->setBatchSize(batchSize);
        Frame::mySegOnnxCV->setInputSize(inputSize);
    }

    if (whether_detect_object)
    {
        det_folder = fSettings["det_folder"].string();
        det_onnx_path = fSettings["det_onnx_path"].string();
        detect_cuboid_obj->myDetectorOnnxRun = std::make_shared<Detector_ONNXRUNTIME>();
        detect_cuboid_obj->myDetectorOnnxCV = std::make_shared<Detector_OpenCV_DNN>();

        detect_cuboid_obj->myDetectorOnnxRun->LoadModel(det_onnx_path);
        detect_cuboid_obj->myDetectorOnnxRun->setClassNames(_classNamesList);
        detect_cuboid_obj->myDetectorOnnxRun->setBatchSize(batchSize);
        detect_cuboid_obj->myDetectorOnnxRun->setInputSize(inputSize);

        detect_cuboid_obj->myDetectorOnnxCV->LoadModel(det_onnx_path);
        detect_cuboid_obj->myDetectorOnnxCV->setClassNames(_classNamesList);
        detect_cuboid_obj->myDetectorOnnxCV->setBatchSize(batchSize);
        detect_cuboid_obj->myDetectorOnnxCV->setInputSize(inputSize);

        bool use_LSD_algorithm = true;
        bool save_to_imgs = false;
        bool save_to_txts = false;
        bool plot_line = false;

        int numOfOctave_ = 1;
        float Octave_ratio = 2.0;  

        detect_cuboid_obj->myLineDetector = std::make_shared<line_lbd_detect>(line_lbd_detect(numOfOctave_, Octave_ratio));
        detect_cuboid_obj->myLineDetector->use_LSD = use_LSD_algorithm;
        detect_cuboid_obj->myLineDetector->line_length_thres = 15;  // remove short edges

        mpByteTracker = std::make_shared<byte_track::BYTETracker>(byte_track::BYTETracker(fps, 30));
        Frame::mpByteTracker = mpByteTracker;
    }
        
    filtered_ground_height = 0;
    first_absolute_scale_frameid = 0;
    first_absolute_scale_framestamp = 0;
    ground_roi_middle = fSettings["ground_roi_middle"];
    ground_roi_lower = fSettings["ground_roi_lower"];
    ground_inlier_pts = fSettings["ground_inlier_pts"];
    ground_dist_ratio = fSettings["ground_dist_ratio"];
    ground_everyKFs = fSettings["ground_everyKFs"];
    camera_object_BA_weight = fSettings["camera_object_BA_weight"];
    object_velocity_BA_weight = fSettings["object_velocity_BA_weight"];

    if (mono_firstframe_truth_depth_init)
    {
        right_folder = fSettings["right_folder"].string();
    }
    
    string scene_name = fSettings["scene_unique_id"].string();
    map<string, Scene_Name> scene_map = {
        {"kitti", ORB_SLAM2::kitti},
        {"kitti_offline", ORB_SLAM2::kitti_offline},
        {"voidtype", ORB_SLAM2::voidtype}
    };

    if (scene_map.count(scene_name))
        scene_unique_id = scene_map[scene_name];
    else
        scene_unique_id = ORB_SLAM2::voidtype;
    
    std::cout << "\nModel Parameters:" << std::endl;
    std::cout << "- scene_unique_id: " << ((scene_unique_id == 0) ? "voidtype" : ((scene_unique_id == 1) ? "kitti" : "kitti_offline") ) << std::endl;
    std::cout << "- build_worldframe_on_ground: " << (build_worldframe_on_ground ? "true" : "false") << std::endl;
    std::cout << "- enable_ground_height_scale: " << (enable_ground_height_scale ? "true" : "false") << std::endl;
    std::cout << "- whether_detect_object: " << (whether_detect_object ? "true" : "false") << std::endl;
    std::cout << "- whether_read_offline_cuboidtxt: " << (whether_read_offline_cuboidtxt ? "true" : "false") << std::endl;
    std::cout << "- associate_point_with_object: " << (associate_point_with_object ? "true" : "false") << std::endl;

    std::cout << "- ba_dyna_pt_obj_cam: " << (ba_dyna_pt_obj_cam ? "true" : "false") << std::endl;
    std::cout << "- ba_dyna_obj_velo: " << (ba_dyna_obj_velo ? "true" : "false") << std::endl;
    std::cout << "- ba_dyna_obj_cam: " << (ba_dyna_obj_cam ? "true" : "false") << std::endl;

    std::cout << "- whether_dynamic_object: " << (whether_dynamic_object ? "true" : "false") << std::endl;
    std::cout << "- remove_dynamic_features: " << (remove_dynamic_features ? "true" : "false") << std::endl;
    std::cout << "- use_dynamic_klt_features: " << (use_dynamic_klt_features ? "true" : "false") << std::endl;
    
    std::cout << "- mono_firstframe_truth_depth_init: " << (mono_firstframe_truth_depth_init ? "true" : "false") << std::endl;
    if (mono_firstframe_truth_depth_init)
    {
        std::cout << "- right_folder: " << right_folder << std::endl;
    }

    std::cout << "- mono_allframe_Obj_depth_init: " << (mono_allframe_Obj_depth_init ? "true" : "false") << std::endl;
    std::cout << "- mono_firstframe_Obj_depth_init: " << (mono_firstframe_Obj_depth_init ? "true" : "false") << std::endl;
    std::cout << "- bundle_object_opti: " << (bundle_object_opti ? "true" : "false") << std::endl;
    std::cout << "- triangulate_dynamic_pts: " << (triangulate_dynamic_pts ? "true" : "false") << std::endl;
    std::cout << "- use_truth_trackid: " << (use_truth_trackid ? "true" : "false") << std::endl;
    std::cout << "- whether_save_online_detected_cuboids: " << (whether_save_online_detected_cuboids ? "true" : "false") << std::endl;
    std::cout << "- whether_save_final_optimized_cuboids: " << (whether_save_final_optimized_cuboids ? "true" : "false") << std::endl;
    std::cout << "- load_truth_pose: " << (load_truth_pose ? "true" : "false") << std::endl;
    if (load_truth_pose)
        std::cout << "- truth_pose_path: " << truth_pose_txts << std::endl;

    std::cout << "- filtered_ground_height: " << filtered_ground_height << std::endl;
    std::cout << "- first_absolute_scale_frameid: " << first_absolute_scale_frameid << std::endl;
    std::cout << "- first_absolute_scale_framestamp: " << first_absolute_scale_framestamp << std::endl;
    std::cout << "- ground_roi_middle: " << ground_roi_middle << std::endl;
    std::cout << "- ground_roi_lower: " << ground_roi_lower << std::endl;
    std::cout << "- ground_inlier_pts: " << ground_inlier_pts << std::endl;
    std::cout << "- ground_dist_ratio: " << ground_dist_ratio << std::endl;
    std::cout << "- ground_everyKFs: " << ground_everyKFs << std::endl;
    std::cout << "- camera_object_BA_weight: " << camera_object_BA_weight << std::endl;
    std::cout << "- object_velocity_BA_weight: " << object_velocity_BA_weight << std::endl;
    std::cout << "- base_data_folder: " << base_data_folder << std::endl;
    std::cout << "- cuboid_folder: " << cuboid_folder << std::endl;
    std::cout << "- seg_folder: " << seg_folder << std::endl;

}

Tracking::~Tracking()
{
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer = pViewer;
}

cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    mImGray = imRectLeft.clone();
    cv::Mat imGrayRight = imRectRight;
    //cv::Mat imBGR;
    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGB2GRAY);
            cvtColor(imRectLeft, mimBGR, cv::COLOR_RGB2BGR);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGR2GRAY);
            mimBGR = imRectLeft;
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_RGBA2GRAY);
            cvtColor(imRectLeft, mimBGR, cv::COLOR_RGBA2BGR);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight, imGrayRight, cv::COLOR_BGRA2GRAY);
            cvtColor(imRectLeft, mimBGR, cv::COLOR_BGRA2BGR);
        }
    }
    else 
    {
        cvtColor(imRectLeft, mimBGR, cv::COLOR_GRAY2BGR);
    }
    // if stereo or RGBD, also set depth and stereo point for it.
    mCurrentFrame = Frame(mImGray, imGrayRight, mimBGR, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, filename);

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB.clone();
    cv::Mat imDepth = imD;
    //cv::Mat imBGR;
    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            cvtColor(imRGB, mimBGR, cv::COLOR_RGB2BGR);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
            mimBGR = imRGB;
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            cvtColor(imRGB, mimBGR, cv::COLOR_RGBA2BGR);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
            cvtColor(imRGB, mimBGR, cv::COLOR_BGRA2BGR);
        }
    }
    else
    {
        cvtColor(imRGB, mimBGR, cv::COLOR_GRAY2BGR);
    }

    if (mDepthMapFactor != 1 || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, imDepth, mimBGR, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, filename);

    if (mCurrentFrame.mnId == 0)
    {
        mpMap->img_height = mImGray.rows;
        mpMap->img_width = mImGray.cols;
    }

    Track();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, int msg_seq_id, string filename)
{
    mImGray = im.clone();
    //cv::Mat imBGR;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGB2GRAY);
            cvtColor(im, mimBGR, cv::COLOR_RGB2BGR);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGR2GRAY);
            mimBGR = im;
        }
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, cv::COLOR_RGBA2GRAY);
            cvtColor(im, mimBGR, cv::COLOR_RGBA2BGR);
        }
        else
        {
            cvtColor(mImGray, mImGray, cv::COLOR_BGRA2GRAY);
            cvtColor(im, mimBGR, cv::COLOR_BGRA2BGR);
        }
    }
    else
    {
        cvtColor(im, mimBGR, cv::COLOR_GRAY2BGR);
    }

    // create frame and detect features!
    if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
    {
        if ((!mono_firstframe_truth_depth_init) || (mCurrentFrame.mnId > 0))
        {
            //std::cout << "using mpIniORBextractor" << std::endl;
            mCurrentFrame = Frame(mImGray, mimBGR, timestamp, mpIniORBextractor, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, filename);
        }
        else
        { // read (truth) depth /stereo image for first frame
            // not good to read first frame's predicted depth by object. quite inaccurate.
            // orb slam will create left/right coordinates based on that, and will be used for optimizer.

            boost::filesystem::path bPath(base_data_folder);

            // Get the parent path
            boost::filesystem::path parentPath = bPath.parent_path();
            // Convert to string
            std::string parentString = parentPath.string();
            //std::cout << "parentString " << parentString << std::endl;

            std::string right_kitti_img_file = parentString + "/" + right_folder + "/" + filename + ".png";
            cv::Mat right_stereo_img = cv::imread(right_kitti_img_file, cv::IMREAD_GRAYSCALE);
            if (!right_stereo_img.data)
                std::cout << "Cannot read first stereo file " << right_kitti_img_file << std::endl;
            else
                std::cout << "Read first right stereo size " << right_stereo_img.rows << std::endl;
            std::cout << "Read first right depth size " << right_stereo_img.rows << " baseline " << mbf << std::endl;
            mCurrentFrame = Frame(mImGray, right_stereo_img, mimBGR, timestamp, mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, filename);
        }
    }
    else
    {
        mCurrentFrame = Frame(mImGray, mimBGR, timestamp, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth, filename); // create new frames.
    }

    if (mCurrentFrame.mnId == 0)
        start_msg_seq_id = msg_seq_id;
    // if read offline txts, frame id must match!!!
    if (all_offline_object_cubes.size() > 0)
    {
        if ((mCurrentFrame.mnId > 0) && (msg_seq_id > 0))                    // if msg_seq_id=0 may because the value is not set.
            if (int(mCurrentFrame.mnId) != (msg_seq_id - start_msg_seq_id)) // can use frame->IdinRawImages = msg_seq_id-start_msg_seq_id  need to change lots of stuff.
            {
                std::cout << "Different frame ID, might due to lost frame from bag. " << mCurrentFrame.mnId << " " << msg_seq_id - start_msg_seq_id << std::endl;
                exit(0);
            }
    }

    if (mCurrentFrame.mnId == 0)
    {
        mpMap->img_height = mImGray.rows;
        mpMap->img_width = mImGray.cols;
    }

    //if (whether_detect_object)
    //{
    //    mCurrentFrame.raw_img = mImGray; // I clone in Keyframe.cc  don't need to clone here.
    //}

    Track(); // main code here

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if (mState == NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState = mState;

    bool created_keyframe = false;

    // Get Map Mutex -> Map cannot be changed
    if (parallel_mapping) // if sequential mapping, no need to use mutex, otherwise conflict with localmapping optimizers.
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED) // initialization
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD)
            StereoInitialization(); // for stereo or RGBD, the first frame is used to initialize the keyframe and map
        else
        {
            bool special_initialization = false;
            //if (mCurrentFrame.mnId == 0)
            //{
                if (mono_firstframe_truth_depth_init)
                {
                    special_initialization = true;
                    StereoInitialization(); // if first frame has truth depth, we can initialize simiar to stereo/rgbd. create keyframe for it.
                }
                else if (mono_firstframe_Obj_depth_init)
                {
                    special_initialization = true;
                    // similar to stereo initialization, but directly create map point. don't create stereo right coordinate
                    // have less effect on g2o optimization.  because depth initialization is not accurate
                    MonoObjDepthInitialization();
                }
            //}
            if (!special_initialization)
                MonocularInitialization(); // usually for monocular, need to wait for several frames, with enough parallax
        }

        mpFrameDrawer->Update(this);

        if (mState != OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless explicitly activate the "only tracking" mode.
            if (mState == OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) // at initialization or relocalization stage
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel(); // usual way of camera tracking
                    if (!bOK)                      // if trackMotion model failed,   usually not happening
                    {
                        bOK = TrackReferenceKeyFrame();
                        if (!bOK)
                            std::cout << "Not OK right after TrackReferenceKeyFrame!!!" << std::endl;
                    }
                }
            }
            else
            {
                std::cout << "Last frame tracking bad, start relocalization!!" << std::endl;
                bOK = Relocalization();
            }
        }
        else // Only Tracking: Local Mapping is deactivated, usually not happening.
        {
            if (mState == LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if (!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if (!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else // not for Mono
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint *> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if (!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if (bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if (mbVO)
                        {
                            for (int i = 0; i < mCurrentFrame.N; i++)
                            {
                                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if (bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if (!mbOnlyTracking) // usually the case
        {
            if (bOK)
            {
                bOK = TrackLocalMap();
                if (!bOK)
                    std::cout << "Not OK right after tracking local map!!!" << std::endl;
            }
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if (bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if (bOK)
            mState = OK;
        else
        {
            std::cout << "Setting tracking state to LOST!!!!!!!!!!!!" << std::endl;
            if (!(whether_detect_object || whether_detect_object))
                mState = LOST; // HACK disable it by me, relocalisation not suitable....
        }

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            // Update motion model
            if (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean temporal point matches  for rgbd/stereo directly inited map point. not monoculari
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (pMP)
                    if (pMP->Observations() < 1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }
            }

            // Delete temporal MapPoints for stere/rgbd
            for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end(); lit != lend; lit++)
            {
                MapPoint *pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame())
            {
                CreateNewKeyFrame(); //push into the localmapping buffer. mapping thread will BA.
                created_keyframe = true;
                // std::cout << "Created new keyframe!   " << mpReferenceKF->mnId << "   total ID  " << mpReferenceKF->mnFrameId << std::endl;
            }

            mpFrameDrawer->Update(this); // I put here so that frame drawer syncs with new keyframe.

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for (int i = 0; i < mCurrentFrame.N; i++)
            {
                if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
            }
        }
        else
        {
            mpFrameDrawer->Update(this);
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST)
        {
            if (mpMap->KeyFramesInMap() <= 5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if (!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse(); //reference to current
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState == LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState == LOST);
    }

    if (whether_save_final_optimized_cuboids)
    {
        if ((mCurrentFrame.mnId >= final_object_record_frame_ind) && (!done_save_obj_to_txt))
        {
            SaveOptimizedCuboidsToTxt();
            done_save_obj_to_txt = true;
            std::cout << "Done save cuboids to txt" << std::endl;
        }
    }

    // implement blocking    block tracking thread, wait until mapping has finished (refine KF)  then track new frame.
    if (!parallel_mapping)
    {
        mpLocalMapper->RunMappingIteration();
    }
}

void Tracking::StereoInitialization()
{
    std::cout << "Stereo initialization!" << std::endl;
    if (mCurrentFrame.N > 500)
    {
        // Set Frame pose to the origin
        if (build_worldframe_on_ground) // transform initial pose and map to ground frame
            mCurrentFrame.SetPose(GroundToInit);
        else
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        // Create KeyFrame.    set (current) first frame as keyframe
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        std::cout << "Created new keyframe! Stereo Init, keyframe.id=" << pKFini->mnId << " frame.id=" << pKFini->mnFrameId << std::endl;

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        double mean_depth = 0;
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0) // monocular has negative value
            {
                mean_depth += z;
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap); // initialize new map points
                pNewMP->AddObservation(pKFini, i);
                pKFini->AddMapPoint(pNewMP, i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i] = pNewMP;
            }
        }

        std::cout << "Mean init frame depth " << mean_depth / (double)mpMap->MapPointsInMap() << std::endl;
        if (mpMap->MapPointsInMap() == 0)
        {
            std::cout << "Zero map point initiated, Wrong!!" << std::endl;
            exit(0);
        }
        std::cout << "New map created with " << mpMap->MapPointsInMap() << " out of all " << mCurrentFrame.N << " feature points" << std::endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState = OK;
    }
    else
        cout << "Not enough points for initialization " << mCurrentFrame.N << std::endl;
}

void Tracking::MonoObjDepthInitialization()
{
    std::cout << "Mono Object depth initialization!" << std::endl;
    //std::cout << "Created new keyframe!   " << 0 << "   total ID  " << 0 << std::endl;
    if (mCurrentFrame.N > 500)
    {
        // Set Frame pose to the origin
        if (build_worldframe_on_ground) // transform initial pose and map to ground frame
            mCurrentFrame.SetPose(GroundToInit);
        else
            mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

        // Create KeyFrame.    set (current) first frame as keyframe
        KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        if (whether_detect_object)
        {
            DetectCuboid(pKFini);
            AssociateCuboids(pKFini); // associate cuboids.  // first frame don't need mpLastKeyFrame as I check .mID==0
        }

        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float point_depth = -1;
            cv::Mat x3D;
            if (whether_detect_object && mono_firstframe_Obj_depth_init && associate_point_with_object)
            {
                int id = pKFini->keypoint_associate_objectID[i];
                if (id > -1)
                {
                    // todo, add build_worldframe_on_ground
                    point_depth = pKFini->local_cuboids[id]->cube_meas.translation()[2]; // camera z
                    x3D = mCurrentFrame.UnprojectDepth(i, point_depth);
                }
                //std::cout << "here, id=" << id << std::endl;
            }

            if (point_depth > 0)
            {
                MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap); // x3d already world frame based on pose
                pKFini->SetupSimpleMapPoints(pNewMP, i);

                mCurrentFrame.mvpMapPoints[i] = pNewMP;
                mCurrentFrame.mvbOutlier[i] = false;
            }
        }

        std::cout << "New map created with " << mpMap->MapPointsInMap() << " out of all " << mCurrentFrame.N << " feature points" << std::endl;

        if (mpMap->MapPointsInMap() == 0)
        {
            std::cout << "Bad MonoObjDepthInitialization! No map points! Break systems!" << std::endl;
            exit(0);
        }

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState = OK;
    }
    else
        cout << "Not enough points for initialization " << mCurrentFrame.N << std::endl;
}

void Tracking::MonocularInitialization()
{
    std::cout << "Normal monocular initialization!" << std::endl;
    if (!mpInitializer)
    {
        // Set Reference Frame
        if (mCurrentFrame.mvKeys.size() > 100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

            if (mpInitializer)
                delete mpInitializer;

            mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if ((int)mCurrentFrame.mvKeys.size() <= 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9, true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100); // 100 is pixel area

        // Check if there are enough correspondences
        if (nmatches < 100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
            return;
        }

        cv::Mat Rcw;                 // Current Camera Rotation
        cv::Mat tcw;                 // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        // call important map initializer here. either homograpy or fundamental
        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    nmatches--;
                }
            }

            // Set Frame Poses   first frame is identity mat
            mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
            Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
            tcw.copyTo(Tcw.rowRange(0, 3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
        else 
        {
            std::cout << "failed to triangle in MonocularInitialization, match " << nmatches << " points" << std::endl;
        }

    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    std::cout << "Created Mono Initial Map using keyframe.id=" << pKFini->mnId << " frame.id=" << pKFini->mnFrameId << std::endl;
    std::cout << "Created Mono Initial Map using keyframe.id=" << pKFcur->mnId << " frame.id=" << pKFcur->mnFrameId << std::endl;

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    if (whether_detect_object)
    {
        DetectCuboid(pKFini);
        AssociateCuboids(pKFini);
        DetectCuboid(pKFcur);
        AssociateCuboids(pKFcur);
    }

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); i++)
    {
        if (mvIniMatches[i] < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        if (mono_allframe_Obj_depth_init)
        {
            if (pKFini->KeysStatic.size() > 0 && !pKFini->KeysStatic[i])
                continue;
            if (pKFcur->KeysStatic.size() > 0 && !pKFcur->KeysStatic[i])
                continue;
        }

        MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20); // TODO whether add object in it... might also make the initialization fail????

    // Set median depth to 1!!!
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f / medianDepth;

    int min_point_num = 50; // byme 100
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < min_point_num)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points. map points are in the first frame, which is world frame. so we can directly scale it
    vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
    {
        if (vpAllMapPoints[iMP])
        {
            MapPoint *pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
        }
    }

    if (build_worldframe_on_ground) // transform initial pose and map to ground frame
    {
        pKFini->SetPose(pKFini->GetPose() * GroundToInit);
        pKFcur->SetPose(pKFcur->GetPose() * GroundToInit);

        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++)
        {
            if (vpAllMapPoints[iMP])
            {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(InitToGround.rowRange(0, 3).colRange(0, 3) * pMP->GetWorldPos() + InitToGround.rowRange(0, 3).col(3));
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose()); //TODO

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose()); //TODO

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.N; i++)
    {
        MapPoint *pMP = mLastFrame.mvpMapPoints[i];

        if (pMP)
        {
            MapPoint *pRep = pMP->GetReplaced();
            if (pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7, true);
    vector<MapPoint *> vpMapPointMatches; // for current frame.

    int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15)
    {
        return false;
    }

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                if (!(whether_dynamic_object && mCurrentFrame.mvpMapPoints[i]->is_dynamic))
                    nmatchesMap++;
        }
    }

    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame *pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR)
        return;

    // Create "visual odometry" MapPoints for stereo/RGBD camera
    // no big use as for RGBD and stereo, as most points can be associated in map.
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float, int>> vDepthIdx; //depth  index
    vDepthIdx.reserve(mLastFrame.N);
    for (int i = 0; i < mLastFrame.N; i++)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
        {
            vDepthIdx.push_back(make_pair(z, i));
        }
    }

    if (vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(), vDepthIdx.end());

    // We insert all 'close' points (depth<mThDepth)
    // If less than 100 'close' points, we insert the 100 closest ones.
    int nPoints = 0;
    int actualCreates = 0;
    for (size_t j = 0; j < vDepthIdx.size(); j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint *pMP = mLastFrame.mvpMapPoints[i];
        if (!pMP)
            bCreateNew = true;
        else if (pMP->Observations() < 1)
        {
            bCreateNew = true;
        }

        if (bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

            mLastFrame.mvpMapPoints[i] = pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
            actualCreates++;
        }
        else
        {
            nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points for RGBD/Stereo.   no use for mono
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // set all matched map points to null, as this is first time to match with map points.
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

    mCurrentFrame.mvpMapPoints_inlastframe.resize(mCurrentFrame.N); // NOTE add by me for visualization

    // Project map points seen in previous frame  onto current frame.
    int searchRadiusFactor;
    if (mSensor != System::STEREO)
        searchRadiusFactor = 15;
    else
        searchRadiusFactor = 7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, searchRadiusFactor, mSensor == System::MONOCULAR);

    // If few matches, uses a wider window search
    if (nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL)); //reset
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2 * searchRadiusFactor, mSensor == System::MONOCULAR);
    }

    if (whether_detect_object)
    {
        if (use_dynamic_klt_features)
            matcher.SearchByTrackingHarris(mCurrentFrame, mLastFrame, searchRadiusFactor, mSensor == System::MONOCULAR);
        else
            matcher.SearchByTracking(mCurrentFrame, mLastFrame, searchRadiusFactor, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20)
    {
        std::cout << "Track with motion model small matches " << nmatches << std::endl;
        return false;
    }

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    mCurrentFrame.mvpMapPoints_lasttracked = mCurrentFrame.mvpMapPoints; // NOTE add by me for visualization

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            if (mCurrentFrame.mvbOutlier[i])
            {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                mCurrentFrame.mvbOutlier[i] = false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                if (!(whether_dynamic_object && mCurrentFrame.mvpMapPoints[i]->is_dynamic))
                    nmatchesMap++;
        }
    }

    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    UpdateLocalMap();

    // find matches with all local map points
    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // std::cout << "track local map marker 4" << std::endl;

    // Update MapPoints Statistics
    int cc = 0;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            cc++;
            if (!mCurrentFrame.mvbOutlier[i])
            {
                if (whether_dynamic_object && mCurrentFrame.mvpMapPoints[i]->is_dynamic)
                    continue;

                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if (!mbOnlyTracking)
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    {
                        mnMatchesInliers++;
                    }
                }
                else
                    mnMatchesInliers++;
            }
            else if (mSensor == System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
        }
    }
    // std::cout << "Current frame points matches:   " << cc << std::endl;

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        return false;

    int map_match_thres = 30;  //NOTE important raw: 30
    if (whether_detect_object) // for dynamic scenarios, there might not be enough feature matches
        map_match_thres = 20;

    if (mnMatchesInliers < map_match_thres)
    {
        std::cout << "Low map num matches inliers " << mnMatchesInliers << std::endl;
        return false;
    }
    else
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if (mbOnlyTracking)
        return false;

    // std::cout << "check whether kf  " << mCurrentFrame.mnId << std::endl;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if (nKFs <= 2)
        nMinObs = 2;
    // if first frame depth initialized by object... not actual measurement. map point only have one observation
    // need to reduce the threshold for initialization.
    if ((mono_firstframe_Obj_depth_init || mono_firstframe_truth_depth_init))
    {
        if ((mSensor == System::MONOCULAR) && (scene_unique_id == kitti || scene_unique_id == kitti_offline) && (mpReferenceKF->mnId < 20)) // for kitti, don't need that many
            nMinObs = 1;
    }
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs); // tracked mappoints with at least nMinObs in reference Key frame

    // whether Local Mapping accept keyframes   during mapping iteration, don't allow to accept keyframes
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
    // "total matches = matches to map + visual odometry matches"
    // Visual odometry matches will become MapPoints if we insert a keyframe.
    // This ratio measures how many MapPoints we could create if we insert a keyframe.
    int nMap = 0;
    int nTotal = 0;
    if (mSensor != System::MONOCULAR)
    {
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
            {
                nTotal++;
                if (mCurrentFrame.mvpMapPoints[i]) // if already associated a map point, done in previous tracking function.
                {
                    if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    {
                        nMap++;
                    }
                }
            }
        }
    }
    else
    {
        // There are no visual odometry matches in the monocular case
        nMap = 1;
        nTotal = 1;
    }

    const float ratioMap = (float)nMap / fmax(1.0f, nTotal);

    // Thresholds
    float thRefRatio = 0.75f;
    if (nKFs < 2)
        thRefRatio = 0.4f; // don't need large threshold at beginning

    if (mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    if (mSensor == System::MONOCULAR)
        if (mono_firstframe_truth_depth_init || mono_firstframe_Obj_depth_init)
            if (nKFs < 2)
                thRefRatio = 0.4f;

    float thMapRatio = 0.35f;
    if (mnMatchesInliers > 300)
        thMapRatio = 0.20f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
    //     const bool c1b = false;
    //Condition 1c: tracking is weak
    const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || ratioMap < 0.3f);
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || ratioMap < thMapRatio) && mnMatchesInliers > 15);
    // usually first cond in c2 is important.

    if ((c1a || c1b || c1c) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if (bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }
}

void Tracking::DetectCuboid(KeyFrame *pKF)
{
    cv::Mat pop_pose_to_ground;              // pop frame pose to ground frame.  for offline txt, usually local ground.  for online detect, usually init ground.
    std::vector<ObjectSet> all_obj_cubes; // in ground frame, no matter read or online detect
    //std::vector<Vector4d> all_obj2d_bbox;
    //std::vector<double> all_box_confidence;
    vector<int> truth_tracklet_ids;

    if (whether_read_offline_cuboidtxt) // saved object txt usually is usually poped in local ground frame, not the global ground frame.
    {
        // need more check
        /*
        if (all_offline_object_cubes.size() == 0)
            return;
        pop_pose_to_ground = InitToGround.clone(); // for kitti, I used InitToGround to pop offline.

        Eigen::MatrixXd pred_frame_objects;
        Eigen::MatrixXd pred_truth_matches;

        //pred_frame_objects = all_offline_object_cubes[(int)pKF->mnFrameId];
        pred_frame_objects = all_offline_object_cubes[pKF->msfilename];
        //Eigen::Matrix4d Twc = Converter::toMatrix4f(pKF->GetPoseInverse()).cast<double>();

        cv::Mat frame_pose_to_init = pKF->GetPoseInverse(); // Twc, camera to init world
        cv::Mat frame_pose_to_ground;
        if (build_worldframe_on_ground){ // if initial world frame is on ground, directly use it.
            frame_pose_to_ground = frame_pose_to_init;
        } else { // if not, apply T_ground_init
            frame_pose_to_ground = InitToGround * frame_pose_to_init;
        }

        Eigen::Matrix4d Twc = Converter::toMatrix4f(frame_pose_to_ground).cast<double>();
        for (int i = 0; i < pred_frame_objects.rows(); i++)
        {
            cuboid *raw_cuboid = new cuboid();
            raw_cuboid->pos = pred_frame_objects.row(i).head(3);
            raw_cuboid->rotY = pred_frame_objects(i, 3);
            //raw_cuboid->scale = Vector3d(pred_frame_objects(i, 4), pred_frame_objects(i, 5), pred_frame_objects(i, 6));
            // https://github.com/lturing/qd_3dt_modified/blob/main/utils.py#L369
            // h, w, l = box_dim, h -> y, w -> z, l -> x
            // default align to the ground
            raw_cuboid->scale = Vector3d(pred_frame_objects(i, 6) / 2.0, pred_frame_objects(i, 4) / 2.0, pred_frame_objects(i, 5) / 2.0);
            raw_cuboid->rect_detect_2d = pred_frame_objects.row(i).segment<4>(7); // will recompute in computeboxes
            
            // computeBoxCorner(raw_cuboid->rotY, raw_cuboid->pos, raw_cuboid->scale, image_width, image_height, Twc, Kalib, raw_cuboid->rect_detect_2d, raw_cuboid->box_corners_2d, raw_cuboid->box_corners_3d_world);
            //std::cout << "rotY=" << raw_cuboid->rotY << ", pos=" << raw_cuboid->pos.transpose() << ", scale=" << raw_cuboid->scale.transpose() << std::endl;
            //std::cout << "before rect_detect_2d=" << raw_cuboid->rect_detect_2d.transpose() << std::endl;
            computeboxes(raw_cuboid, Twc, Kalib, image_width, image_height);
            //std::cout << "after rect_detect_2d=" << raw_cuboid->rect_detect_2d.transpose() << std::endl;
            raw_cuboid->box_config_type = Vector2d(1, 1); // randomly given unless provided. for latter visualization

            // #todo
            raw_cuboid->confidence = pred_frame_objects(i, 11);
            raw_cuboid->classId = 1;

            //all_obj2d_bbox.push_back(raw_cuboid->rect_detect_2d);
            //all_box_confidence.push_back(pred_frame_objects(i, 11));
            if (use_truth_trackid)
                truth_tracklet_ids.push_back(pred_frame_objects(i, 12));
            ObjectSet temp;
            temp.push_back(raw_cuboid);
            all_obj_cubes.push_back(temp);
        }

        */
    }
    else
    {
        Eigen::MatrixXd raw_all_obj2d_bbox; // left_top_x, left_top_y, width, height, confidence, classId
        if (! whether_dynamic_object)
        {
            if (1)
            {
                cv::Mat img = pKF->mImBGR.clone();
                //if( img.channels() == 1 )
                //    cvtColor(img, img, cv::COLOR_GRAY2BGR);
                
                std::vector<cv::Mat> imgBatch;
                //imgBatch.clear();
                imgBatch.push_back(img);
                BatchDetectedObject result;
                #if USE_ONNX
                    result = detect_cuboid_obj->myDetectorOnnxRun->Run(imgBatch);
                #else 
                    result = detect_cuboid_obj->myDetectorOnnxCV->Run(imgBatch);
                #endif 
                
                use_truth_trackid = false; // track_id is not available
                raw_all_obj2d_bbox.resize(result[0].size(), 7);
                for(int i = 0; i < result[0].size(); i++)
                {
                    // [x y w h prob]
                    raw_all_obj2d_bbox(i, 0) = result[0][i].box.tl().x;
                    raw_all_obj2d_bbox(i, 1) = result[0][i].box.tl().y;
                    raw_all_obj2d_bbox(i, 2) = result[0][i].box.size().width; //w;
                    raw_all_obj2d_bbox(i, 3) = result[0][i].box.size().height; // h;
                    raw_all_obj2d_bbox(i, 4) = result[0][i].confidence;
                    raw_all_obj2d_bbox(i, 5) = result[0][i].classID;
                    raw_all_obj2d_bbox(i, 6) = 0; // track_id, not use
                }
                result.clear();
            }
            else 
            {
                // read from offline 
                // for offline, track_id can be available 
                std::string pred_info_name = base_data_folder + "/" + det_folder + "/" + pKF->msfilename + ".txt";
                if(!read_obj_yolo_txt(pred_info_name, raw_all_obj2d_bbox))
                {
                    std::cout << "Cannot read yolo txt " << pred_info_name << std::endl;
                    return ;
                }
            }
        }
        else 
        {
            raw_all_obj2d_bbox = pKF->raw_all_obj2d_bbox;
        }

        // remove some 2d boxes too close to boundary.
        int boundary_threshold = 10;
        int img_width = pKF->mImBGR.cols;
        int img_height = pKF->mImBGR.rows;

        std::vector<int> good_object_ids;
        for (int i = 0; i < raw_all_obj2d_bbox.rows(); i++)
        {
            if ((raw_all_obj2d_bbox(i, 0) > boundary_threshold) && (raw_all_obj2d_bbox(i, 0) + raw_all_obj2d_bbox(i, 2) < img_width - boundary_threshold) &&
                (raw_all_obj2d_bbox(i, 1) > boundary_threshold) && (raw_all_obj2d_bbox(i, 1) + raw_all_obj2d_bbox(i, 3) < img_height - boundary_threshold))
            {
                int classId = int(raw_all_obj2d_bbox(i, 5));
                std::string className = detect_cuboid_obj->myDetectorOnnxRun->getClassName(classId);
                if (className == "car")
                {
                    good_object_ids.push_back(i);
                } 
            }
            //else 
            //    std::cout << "raw_all_obj2d_bbox(i)=" << raw_all_obj2d_bbox.row(i) << ", img_width=" << img_width << ", img_height=" << img_height << std::endl;
        }

        Eigen::MatrixXd all_obj2d_bbox_infov_mat(good_object_ids.size(), 7);

        for (unsigned long i = 0; i < good_object_ids.size(); i++)
        {
            // top_left_x, top_left_y, width, height, confidence, classid, track_id
            all_obj2d_bbox_infov_mat.row(i) = raw_all_obj2d_bbox.row(good_object_ids[i]);
        }

        //if (good_object_ids.size() < 1)
        //{
        //    std::cout << "no object in frame" << std::endl;
        //    return ;
        //}

        //Eigen::MatrixXd all_obj2d_bbox_infov_mat(good_object_ids.size(), 6);
        //for (size_t i = 0; i < good_object_ids.size(); i++)
        //{
        //    all_obj2d_bbox_infov_mat.row(i) = raw_all_obj2d_bbox.row(good_object_ids[i]);
        //}
        
        //std::string data_edge_data_dir = base_data_folder + "/edge_detection/LSD/";
        //std::string data_yolo_obj_dir = base_data_folder + "/mats/filter_match_2d_boxes_txts/";
        //char frame_index_c[256];
        //sprintf(frame_index_c, "%04d", (int)pKF->mnFrameId); // format into 4 digit

        // read detected edges
        //Eigen::MatrixXd all_lines_raw(100, 4); // 100 is some large frame number,   the txt edge index start from 0
        //read_all_number_txt(data_edge_data_dir + frame_index_c + "_edge.txt", all_lines_raw);

        // detect line
        Eigen::MatrixXd all_lines_raw; // 100 is some large frame number,   the txt edge index start from 0
        all_lines_raw.resize(100, 4);
        // using my line detector class, could select LSD or edline.
        cv::Mat out_edges;
        std::vector< KeyLine> keylines_raw, keylines_out;
        detect_cuboid_obj->myLineDetector->detect_raw_lines(pKF->mImBGR, keylines_raw);
        detect_cuboid_obj->myLineDetector->filter_lines(keylines_raw, keylines_out);  // remove short lines

        int row_counter = 0;
        for (int i = 0; i < keylines_out.size(); i++)
        {
            all_lines_raw(i, 0) = keylines_out[i].startPointX;
            all_lines_raw(i, 1) = keylines_out[i].startPointY;
            all_lines_raw(i, 2) = keylines_out[i].endPointX;
            all_lines_raw(i, 3) = keylines_out[i].endPointY;

            row_counter += 1;
            if (row_counter >= all_lines_raw.rows()) // if matrix row is not enough, make more space.
                all_lines_raw.conservativeResize(all_lines_raw.rows() * 2, all_lines_raw.cols());
        }

        all_lines_raw.conservativeResize(row_counter, all_lines_raw.cols()); // cut into actual rows
        
        cv::Mat frame_pose_to_init = pKF->GetPoseInverse(); // Twc, camera to init world
        cv::Mat frame_pose_to_ground;
        if (build_worldframe_on_ground){ // if initial world frame is on ground, directly use it.
            frame_pose_to_ground = frame_pose_to_init;
        } else { // if not, apply T_ground_init
            frame_pose_to_ground = InitToGround * frame_pose_to_init;
        }

        pop_pose_to_ground = frame_pose_to_ground;
        Eigen::Matrix4f cam_transToGround = Converter::toMatrix4f(pop_pose_to_ground);
        detect_cuboid_obj->detect_cuboid(pKF->mImGray, cam_transToGround.cast<double>(), all_obj2d_bbox_infov_mat, all_lines_raw, all_obj_cubes);

        std::cout << "detect " << all_obj_cubes.size() << " 3d cuboids" << std::endl;
    }

    // copy and analyze results. change to g2o cuboid.
    pKF->local_cuboids.clear();
    g2o::SE3Quat frame_pose_to_init = Converter::toSE3Quat(pKF->GetPoseInverse()); // Twc, camera to init, not always ground.
    g2o::SE3Quat InitToGround_se3 = Converter::toSE3Quat(InitToGround);
    for (int ii = 0; ii < (int)all_obj_cubes.size(); ii++)
    {
        if (all_obj_cubes[ii].size() > 0) // if has detected 3d Cuboid
        {
            cuboid *raw_cuboid = all_obj_cubes[ii][0];
            // measurement in local camera frame! important
            MapObject *newcuboid = new MapObject(mpMap);
            g2o::cuboid cube_local_meas; // offline cuboid txt in local ground frame.  [x y z yaw l w h]
            g2o::cuboid global_obj_pose_to_init;
            if (whether_read_offline_cuboidtxt)
            {
                // need more check
                /*
                Vector9d cube_pose;
                cube_pose << raw_cuboid->pos[0], raw_cuboid->pos[1], raw_cuboid->pos[2], 0, raw_cuboid->rotY, 0,
                    raw_cuboid->scale[0], raw_cuboid->scale[1], raw_cuboid->scale[2];
                cube_local_meas.fromMinimalVector(cube_pose); // Tco

                newcuboid->cube_meas = cube_local_meas;
                newcuboid->bbox_2d = cv::Rect(raw_cuboid->rect_detect_2d[0], raw_cuboid->rect_detect_2d[1], raw_cuboid->rect_detect_2d[2], raw_cuboid->rect_detect_2d[3]);
                newcuboid->bbox_vec = Vector4d((double)newcuboid->bbox_2d.x + (double)newcuboid->bbox_2d.width / 2, (double)newcuboid->bbox_2d.y + (double)newcuboid->bbox_2d.height / 2,
                                                (double)newcuboid->bbox_2d.width, (double)newcuboid->bbox_2d.height);
                newcuboid->box_corners_2d = raw_cuboid->box_corners_2d;
                newcuboid->bbox_2d_tight = cv::Rect(raw_cuboid->rect_detect_2d[0] + raw_cuboid->rect_detect_2d[2] / 10.0,
                                                    raw_cuboid->rect_detect_2d[1] + raw_cuboid->rect_detect_2d[3] / 10.0,
                                                    raw_cuboid->rect_detect_2d[2] * 0.8, raw_cuboid->rect_detect_2d[3] * 0.8);
                get_cuboid_draw_edge_markers(newcuboid->edge_markers, raw_cuboid->box_config_type, false);
                newcuboid->SetReferenceKeyFrame(pKF);
                newcuboid->object_id_in_localKF = pKF->local_cuboids.size();

                //g2o::cuboid global_obj_pose_to_init = cube_local_meas.transform_from(frame_pose_to_init);
                global_obj_pose_to_init = cube_local_meas.transform_from(frame_pose_to_init);
                newcuboid->SetWorldPos(global_obj_pose_to_init);
                newcuboid->pose_noopti = global_obj_pose_to_init;
                */
            }
            else 
            {
                g2o::cuboid cube_ground_value; // offline cuboid txt in local ground frame.  [x y z yaw l w h]
                Vector9d cube_pose;
                cube_pose << raw_cuboid->pos[0], raw_cuboid->pos[1], raw_cuboid->pos[2], 0, 0, raw_cuboid->rotY,
                    raw_cuboid->scale[0], raw_cuboid->scale[1], raw_cuboid->scale[2];
                cube_ground_value.fromMinimalVector(cube_pose); // Two

                // measurement in local camera frame! important
                //g2o::cuboid cube_local_meas = cube_ground_value.transform_to(Converter::toSE3Quat(pop_pose_to_ground)); // Tco, object to camera
                cube_local_meas = cube_ground_value.transform_to(Converter::toSE3Quat(pop_pose_to_ground)); // Tco, object to camera
                newcuboid->cube_meas = cube_local_meas;
                newcuboid->bbox_2d = cv::Rect(raw_cuboid->rect_detect_2d[0], raw_cuboid->rect_detect_2d[1], raw_cuboid->rect_detect_2d[2], raw_cuboid->rect_detect_2d[3]);
                newcuboid->bbox_vec = Vector4d((double)newcuboid->bbox_2d.x + (double)newcuboid->bbox_2d.width / 2, (double)newcuboid->bbox_2d.y + (double)newcuboid->bbox_2d.height / 2,
                                                (double)newcuboid->bbox_2d.width, (double)newcuboid->bbox_2d.height);
                newcuboid->box_corners_2d = raw_cuboid->box_corners_2d;
                newcuboid->bbox_2d_tight = cv::Rect(raw_cuboid->rect_detect_2d[0] + raw_cuboid->rect_detect_2d[2] / 10.0,
                                                    raw_cuboid->rect_detect_2d[1] + raw_cuboid->rect_detect_2d[3] / 10.0,
                                                    raw_cuboid->rect_detect_2d[2] * 0.8, raw_cuboid->rect_detect_2d[3] * 0.8);
                get_cuboid_draw_edge_markers(newcuboid->edge_markers, raw_cuboid->box_config_type, false);
                newcuboid->SetReferenceKeyFrame(pKF);
                newcuboid->object_id_in_localKF = pKF->local_cuboids.size();

                //g2o::cuboid global_obj_pose_to_init = cube_local_meas.transform_from(frame_pose_to_init);
                global_obj_pose_to_init = cube_local_meas.transform_from(frame_pose_to_init);

                newcuboid->SetWorldPos(global_obj_pose_to_init);
                newcuboid->pose_noopti = global_obj_pose_to_init;
            }

            if (use_truth_trackid)
            {
                newcuboid->truth_tracklet_id = raw_cuboid->trackId;
                //newcuboid->truth_tracklet_id = truth_tracklet_ids[ii];
            }

            if (whether_dynamic_object)
            {
                newcuboid->is_dynamic = true;                            // for debug, later should check!
                newcuboid->pose_Twc_latestKF = global_obj_pose_to_init; //set pose for dynamic object
            }
            if (scene_unique_id == kitti)
            {
                if (cube_local_meas.pose.translation()(0) > 1)
                    newcuboid->left_right_to_car = 2; // right
                if (cube_local_meas.pose.translation()(0) < -1)
                    newcuboid->left_right_to_car = 1; // left
                if ((cube_local_meas.pose.translation()(0) > -1) && (cube_local_meas.pose.translation()(0) < 1))
                    newcuboid->left_right_to_car = 0;
            }
            if (1)
            {
                double obj_cam_dist = std::min(std::max(newcuboid->cube_meas.translation()(2), 10.0), 30.0); // cut into [a,b]
                double obj_meas_quality = (60.0 - obj_cam_dist) / 40.0;
                newcuboid->meas_quality = obj_meas_quality;
            }
            else
                newcuboid->meas_quality = 1.0;
            
            //if (all_box_confidence[ii] > 0)
            //    newcuboid->meas_quality *= all_box_confidence[ii]; // or =
            newcuboid->meas_quality *= raw_cuboid->confidence;

            if (newcuboid->meas_quality < 0.1)
                std::cout << "Abnormal measure quality!!: " << newcuboid->meas_quality << std::endl;
            pKF->local_cuboids.push_back(newcuboid);
        }
    }

    //std::cout << "created local object num   " << pKF->local_cuboids.size() << std::endl;
    //std::cout << "Detect cuboid for pKF id: " << pKF->mnId << "  total id: " << pKF->mnFrameId << "  numObj: " << pKF->local_cuboids.size() << std::endl;

    if (whether_save_online_detected_cuboids && !whether_read_offline_cuboidtxt)
    {
        for (int ii = 0; ii < (int)all_obj_cubes.size(); ii++)
        {
            if (all_obj_cubes[ii].size() > 0) // if has detected 3d Cuboid, always true in this case
            {
                cuboid *raw_cuboid = all_obj_cubes[ii][0];
                g2o::cuboid cube_ground_value;
                Vector9d cube_pose;
                cube_pose << raw_cuboid->pos[0], raw_cuboid->pos[1], raw_cuboid->pos[2], 0, 0, raw_cuboid->rotY,
                    raw_cuboid->scale[0], raw_cuboid->scale[1], raw_cuboid->scale[2];
                save_online_detected_cuboids << pKF->mnFrameId << "  " << cube_pose.transpose() << "\n";
            }
        }
    }

    if (associate_point_with_object)
    {
        if (!whether_dynamic_object) //for old non-dynamic object, associate based on 2d overlap... could also use instance segmentation
        {
            pKF->keypoint_associate_objectID = vector<int>(pKF->mvKeys.size(), -1);
            std::vector<bool> overlapped(pKF->local_cuboids.size(), false);
            if (1)
            {
                for (size_t i = 0; i < pKF->local_cuboids.size(); i++)
                    if (!overlapped[i])
                        for (size_t j = i + 1; j < pKF->local_cuboids.size(); j++)
                            if (!overlapped[j])
                            {
                                float iou_ratio = bboxOverlapratio(pKF->local_cuboids[i]->bbox_2d, pKF->local_cuboids[j]->bbox_2d);
                                if (iou_ratio > 0.15)
                                {
                                    overlapped[i] = true;
                                    overlapped[j] = true;
                                }
                            }
            }

            if (!enable_ground_height_scale)
            {                                       // slightly faster
                if (pKF->local_cuboids.size() > 0) // if there is object
                    for (size_t i = 0; i < pKF->mvKeys.size(); i++)
                    {
                        int associated_times = 0;
                        for (size_t j = 0; j < pKF->local_cuboids.size(); j++)
                            if (!overlapped[j])
                                if (pKF->local_cuboids[j]->bbox_2d.contains(pKF->mvKeys[i].pt))
                                {
                                    associated_times++;
                                    if (associated_times == 1)
                                        pKF->keypoint_associate_objectID[i] = j;
                                    else
                                        pKF->keypoint_associate_objectID[i] = -1;
                                }
                                else 
                                {
                                    //std::cout << "not in, bbox_2d=" << pKF->local_cuboids[j]->bbox_2d << std::endl;
                                }
                    }
            }
            else
            {
                pKF->keypoint_inany_object = vector<bool>(pKF->mvKeys.size(), false);
                for (size_t i = 0; i < pKF->mvKeys.size(); i++)
                {
                    int associated_times = 0;
                    for (size_t j = 0; j < pKF->local_cuboids.size(); j++)
                        if (pKF->local_cuboids[j]->bbox_2d.contains(pKF->mvKeys[i].pt))
                        {
                            pKF->keypoint_inany_object[i] = true;
                            if (!overlapped[j])
                            {
                                associated_times++;
                                if (associated_times == 1)
                                    pKF->keypoint_associate_objectID[i] = j;
                                else
                                    pKF->keypoint_associate_objectID[i] = -1;
                            }
                        }
                }
                if (height_esti_history.size() == 0)
                {
                    pKF->local_cuboids.clear(); // don't do object when in initial stage...
                    pKF->keypoint_associate_objectID.clear();
                }
            }
        }

        if (whether_dynamic_object) //  for dynamic object, I use instance segmentation
        {
            if (pKF->local_cuboids.size() > 0) // if there is object
            {
                std::vector<MapPoint *> framePointMatches = pKF->GetMapPointMatches();

                if (pKF->keypoint_associate_objectID.size() < pKF->mvKeys.size())
                    std::cout << "Tracking Bad keypoint associate ID size " << pKF->keypoint_associate_objectID.size() << " " << pKF->mvKeys.size() << std::endl;

                for (size_t i = 0; i < pKF->mvKeys.size(); i++)
                {
                    if (pKF->keypoint_associate_objectID[i] >= 0 && pKF->keypoint_associate_objectID[i] >= pKF->local_cuboids.size())
                    {
                        std::cout << "Detect cuboid find bad pixel obj id " << pKF->keypoint_associate_objectID[i] << " " << pKF->local_cuboids.size() << std::endl;
                    }
                    if (pKF->keypoint_associate_objectID[i] > -1)
                    {
                        MapPoint *pMP = framePointMatches[i];
                        if (pMP)
                            pMP->is_dynamic = true;
                    }
                }
            }
        }
        
    }

    std::vector<KeyFrame *> checkframes = mvpLocalKeyFrames; // only check recent to save time

    int object_own_point_threshold = 20;
    if (scene_unique_id == kitti || scene_unique_id == kitti_offline)
    {
        if (mono_allframe_Obj_depth_init)
            object_own_point_threshold = 50; // 50 using 10 is too noisy.... many objects don't have enough points to match with others then will create as new...
        else
            object_own_point_threshold = 30; // 30 if don't initialize object point sepratedly, there won't be many points....  tried 20, not good...
    }

    if (whether_dynamic_object)
    {
        if (mono_allframe_Obj_depth_init)
            object_own_point_threshold = 10;
    }

    if (use_truth_trackid) //very accurate, no need of object point for association
        object_own_point_threshold = -1;

    // points and object are related in local mapping, when creating mapPoints

    //dynamic object: didn't triangulate point in localmapping. but in tracking
    for (size_t i = 0; i < checkframes.size(); i++)
    {
        KeyFrame *kfs = checkframes[i];
        for (size_t j = 0; j < kfs->local_cuboids.size(); j++)
        {
            MapObject *mPO = kfs->local_cuboids[j];
            if (!mPO->become_candidate)
            {
                // points number maybe increased when later triangulated
                mPO->check_whether_valid_object(object_own_point_threshold);
            }
        }
    }
}

void Tracking::AssociateCuboids(KeyFrame *pKF)
{
    // loop over current KF's objects, check with all past objects (or local objects), compare the associated object map points.
    // (if a local object is not associated, could re-check here as frame-object-point might change overtime, especially due to triangulation.)

    std::vector<MapObject *> LocalObjectsCandidates;
    std::vector<MapObject *> LocalObjectsLandmarks;
    // keypoint might not added to frame observation yet, so object might not have many associated points yet....
    // method 1: just check current frame's object, using existing map point associated objects.
    // same as plane association, don't just check current frame, but check all recent keyframe's unmatched objects...
    for (size_t i = 0; i < mvpLocalKeyFrames.size(); i++) // pKF is not in mvpLocalKeyFrames yet
    {
        KeyFrame *kfs = mvpLocalKeyFrames[i];
        for (size_t j = 0; j < kfs->local_cuboids.size(); j++)
        {
            MapObject *mPO = kfs->local_cuboids[j];
            if (mPO->become_candidate && (!mPO->already_associated))
                LocalObjectsCandidates.push_back(kfs->local_cuboids[j]);
        }
        for (size_t j = 0; j < kfs->cuboids_landmark.size(); j++)
            if (kfs->cuboids_landmark[j]) // might be deleted due to badFlag()
                if (!kfs->cuboids_landmark[j]->isBad())
                    if (kfs->cuboids_landmark[j]->association_refid_in_tracking != pKF->mnId) // could also use set to avoid duplicates
                    {
                        LocalObjectsLandmarks.push_back(kfs->cuboids_landmark[j]);
                        kfs->cuboids_landmark[j]->association_refid_in_tracking = pKF->mnId;
                    }
    }

    //std::cout << "begin to associate cuboids #candidate:   " << LocalObjectsCandidates.size() << "   #landmarks   " << LocalObjectsLandmarks.size() << "   #localKFs   " << mvpLocalKeyFrames.size() << std::endl;
    
    int largest_shared_num_points_thres = 10;
    if (mono_allframe_Obj_depth_init)
        largest_shared_num_points_thres = 20;
    if (scene_unique_id == kitti || scene_unique_id == kitti_offline)
        largest_shared_num_points_thres = 10; // kitti vehicle occupy large region

    if (whether_detect_object && mono_allframe_Obj_depth_init) // dynamic object is more difficult. especially reverse motion
        largest_shared_num_points_thres = 5;

    MapObject *last_new_created_object = nullptr;
    for (size_t i = 0; i < LocalObjectsCandidates.size(); i++)
    {
        // there might be some new created object!
        if (last_new_created_object)
            LocalObjectsLandmarks.push_back(last_new_created_object);
        last_new_created_object = nullptr;

        // find existing object landmarks which share most points with this object
        MapObject *candidateObject = LocalObjectsCandidates[i];
        std::vector<MapPoint *> object_owned_pts = candidateObject->GetPotentialMapPoints();

        MapObject *largest_shared_objectlandmark = nullptr;
        if (LocalObjectsLandmarks.size() > 0)
        {
            map<MapObject *, int> LandmarkObserveCounter;

            for (size_t j = 0; j < object_owned_pts.size(); j++)
                for (map<MapObject *, int>::iterator mit = object_owned_pts[j]->MapObjObservations.begin(); mit != object_owned_pts[j]->MapObjObservations.end(); mit++)
                    LandmarkObserveCounter[mit->first]++;

            int largest_shared_num_points = largest_shared_num_points_thres;
            for (size_t j = 0; j < LocalObjectsLandmarks.size(); j++)
            {
                MapObject *pMP = LocalObjectsLandmarks[j];
                if (!pMP->isBad())
                    if (LandmarkObserveCounter.count(pMP))
                    {
                        if (LandmarkObserveCounter[pMP] > largest_shared_num_points)
                        {
                            largest_shared_num_points = LandmarkObserveCounter[pMP];
                            largest_shared_objectlandmark = pMP;
                        }
                    }
            }
        }

        if (use_truth_trackid) // find associate id based on tracket id.
        {
            if (trackletid_to_landmark.count(candidateObject->truth_tracklet_id))
                largest_shared_objectlandmark = trackletid_to_landmark[candidateObject->truth_tracklet_id];
            else
                largest_shared_objectlandmark == nullptr;
        }

        if (largest_shared_objectlandmark == nullptr) // if not found, create as new landmark.  either using original local pointer, or initialize as new
        {
            if (use_truth_trackid)
            {
                if (candidateObject->truth_tracklet_id > -1) // -1 means no ground truth tracking ID, don't use this object
                    trackletid_to_landmark[candidateObject->truth_tracklet_id] = candidateObject;
                else
                    continue;
            }
            candidateObject->already_associated = true; // must be put before SetAsLandmark();
            KeyFrame *refframe = candidateObject->GetReferenceKeyFrame();
            candidateObject->addObservation(refframe, candidateObject->object_id_in_localKF); // add to frame observation
            refframe->cuboids_landmark.push_back(candidateObject);
            candidateObject->mnId = MapObject::getIncrementedIndex(); //mpMap->MapObjectsInMap();  // needs to manually set
            candidateObject->associated_landmark = candidateObject;
            candidateObject->SetAsLandmark();
            if (scene_unique_id == kitti) // object scale change back and forth
            {
                g2o::cuboid cubeglobalpose = candidateObject->GetWorldPos();
                cubeglobalpose.setScale(Eigen::Vector3d(1.9420, 0.8143, 0.7631));
                candidateObject->SetWorldPos(cubeglobalpose);
                candidateObject->pose_Twc_latestKF = cubeglobalpose;
                candidateObject->pose_noopti = cubeglobalpose;
                candidateObject->allDynamicPoses[refframe] = make_pair(cubeglobalpose, false); //Vector6d::Zero()  false means not BAed
            }
            else if (scene_unique_id == kitti_offline)
            {
                g2o::cuboid cubeglobalpose = candidateObject->GetWorldPos();
                cubeglobalpose.setScale(candidateObject->cube_meas.scale); //Eigen::Vector3d(1.9420, 0.8143, 0.7631));
                candidateObject->SetWorldPos(cubeglobalpose);
                candidateObject->pose_Twc_latestKF = cubeglobalpose;
                candidateObject->pose_noopti = cubeglobalpose;
                candidateObject->allDynamicPoses[refframe] = make_pair(cubeglobalpose, false); //Vector6d::Zero()  false means not BAed
            }
            mpMap->AddMapObject(candidateObject);
            last_new_created_object = candidateObject;
            candidateObject->allDynamicPoses[refframe] = make_pair(candidateObject->GetWorldPos(), false);
        }
        else // if found, then update observation.
        {
            candidateObject->already_associated = true; // must be put before SetAsLandmark();
            KeyFrame *refframe = candidateObject->GetReferenceKeyFrame();
            largest_shared_objectlandmark->addObservation(refframe, candidateObject->object_id_in_localKF);
            refframe->cuboids_landmark.push_back(largest_shared_objectlandmark);
            candidateObject->associated_landmark = largest_shared_objectlandmark;

            //NOTE use current frame's object poes, but don't use current object if very close to boundary.... large error
            // I use this mainly for kitti, as further objects are inaccurate.  for indoor object, we may not need it
            if (scene_unique_id == kitti)
            {
                g2o::cuboid cubeglobalpose = candidateObject->GetWorldPos();
                cubeglobalpose.setScale(Eigen::Vector3d(1.9420, 0.8143, 0.7631));

                largest_shared_objectlandmark->allDynamicPoses[refframe] = make_pair(cubeglobalpose, false);
                largest_shared_objectlandmark->SetWorldPos(cubeglobalpose);
                largest_shared_objectlandmark->pose_Twc_latestKF = cubeglobalpose; //if want to test without BA
                largest_shared_objectlandmark->pose_noopti = cubeglobalpose;
            }
            else if (scene_unique_id == kitti_offline)
            {
                g2o::cuboid cubeglobalpose = candidateObject->GetWorldPos();
                cubeglobalpose.setScale(candidateObject->cube_meas.scale); //Eigen::Vector3d(1.9420, 0.8143, 0.7631));
                
                largest_shared_objectlandmark->allDynamicPoses[refframe] = make_pair(cubeglobalpose, false);
                largest_shared_objectlandmark->SetWorldPos(cubeglobalpose);
                largest_shared_objectlandmark->pose_Twc_latestKF = cubeglobalpose; //if want to test without BA
                largest_shared_objectlandmark->pose_noopti = cubeglobalpose;
            }
            largest_shared_objectlandmark->MergeIntoLandmark(candidateObject);
        }
    }

    // remove outlier objects....
    bool remove_object_outlier = true;

    int minimum_object_observation = 2;
    if (scene_unique_id == kitti || scene_unique_id == kitti_offline)
    {
        remove_object_outlier = false;
        if (whether_detect_object)
        {
            remove_object_outlier = true;
            minimum_object_observation = 3; // dynamic object has more outliers
        }
    }

    bool check_object_points = true;

    if (remove_object_outlier)
    {
        vector<MapObject *> all_objects = mpMap->GetAllMapObjects();
        for (size_t i = 0; i < all_objects.size(); i++)
        {
            MapObject *pMObject = all_objects[i];
            if ((!pMObject->isBad()) && (!pMObject->isGood))                        // if not determined good or bad yet.
                if ((int)pMObject->GetLatestKeyFrame()->mnId < (int)pKF->mnId - 15) //20
                {
                    // if not recently observed, and not enough observations.  NOTE if point-object not used in BA, filtered size will be zero...
                    bool no_enough_inlier_pts = check_object_points && (pMObject->NumUniqueMapPoints() > 20) && (pMObject->used_points_in_BA_filtered.size() < 10) && (pMObject->point_object_BA_counter > -1);
                    if (pMObject->Observations() < minimum_object_observation)
                    {
                        pMObject->SetBadFlag();
                        //cout << "Found one bad object !!!!!!!!!!!!!!!!!!!!!!!!!  " << pMObject->mnId << "  " << pMObject->Observations() << "  " << pMObject->used_points_in_BA_filtered.size() << endl;

                        if (use_truth_trackid)
                            trackletid_to_landmark.erase(pMObject->truth_tracklet_id); // remove from track id mapping
                    }
                    else
                    {
                        pMObject->isGood = true;
                    }
                }
        }
    }
}

template <class BidiIter> //Fisher-Yates shuffle
BidiIter random_unique2(BidiIter begin, BidiIter end, int num_random)
{
    size_t left = std::distance(begin, end);
    while (num_random--)
    {
        BidiIter r = begin;
        std::advance(r, rand() % left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

void Tracking::CreateNewKeyFrame()
{
    if (!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    //std::cout << "Created new keyframe!   " << pKF->mnId << "   total ID  " << pKF->mnFrameId << std::endl;

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (whether_detect_object)
    {
        DetectCuboid(pKF);
        AssociateCuboids(pKF);
    }

    // for stereo/RGBD sensor, also create map keypoints from current frame.
    if (mSensor != System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth. why??
        // If there are more than 100 close points we create the 100 closest.
        vector<pair<float, int>> vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for (int i = 0; i < mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if (z > 0)
            {
                vDepthIdx.push_back(make_pair(z, i));
            }
        }

        if (!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(), vDepthIdx.end()); // from close depth to far

            int nPoints = 0;
            int actualCreates = 0;
            for (size_t j = 0; j < vDepthIdx.size(); j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP)
                    bCreateNew = true;
                else if (pMP->Observations() < 1)
                {
                    // should not happen.... unless the associated mappoint is bad?
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                if (bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap); // also initialize point global position
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                    nPoints++;
                    actualCreates++;
                }
                else
                {
                    nPoints++;
                }
                if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                    break;
            }
        }
    }

    //copied from localMapping, only for dynamic object
    if (mono_allframe_Obj_depth_init && whether_dynamic_object)
    {
        KeyFrame *mpCurrentKeyFrame = pKF;

        const vector<MapPoint *> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

        double total_feature_pts = vpMapPointMatches.size();
        double raw_depth_pts = 0; // point already have depth
        double plane_object_initialized_pts = 0;
        std::vector<int> raw_pixels_no_depth_inds;

        if (triangulate_dynamic_pts)
        {
            vector<MapPoint *> frameMapPointMatches;
            if (use_dynamic_klt_features)
                frameMapPointMatches = mCurrentFrame.mvpMapPointsHarris;
            else
                frameMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

            cv::Mat Tcw_last = mpLastKeyFrame->GetPose();
            cv::Mat Tcw_now = mpCurrentKeyFrame->GetPose();
            const float &cx1 = mpCurrentKeyFrame->cx;
            const float &cy1 = mpCurrentKeyFrame->cy;
            const float &invfx1 = mpCurrentKeyFrame->invfx;
            const float &invfy1 = mpCurrentKeyFrame->invfy;
            for (size_t i = 0; i < frameMapPointMatches.size(); i++)
            {
                MapPoint *pMP = frameMapPointMatches[i];
                if (pMP && pMP->is_dynamic) // the point is matched to this frame, and also dynamic.
                {
                    // check if this point is created by last keyframe, if yes, triangulate it with this frame!   if created earlier, not need
                    if (!pMP->is_triangulated) // if not Triangulated
                    {
                        int pixelindLastKf = pMP->GetIndexInKeyFrame(mpLastKeyFrame);
                        if (pixelindLastKf == -1)
                        {
                            std::cout << "Point frame observation not added yet" << std::endl;
                            continue;
                        }
                        MapObject *objectLastframe;
                        if (use_dynamic_klt_features)
                            objectLastframe = mpLastKeyFrame->local_cuboids[mpLastKeyFrame->keypoint_associate_objectID_harris[pixelindLastKf]];
                        else
                            objectLastframe = mpLastKeyFrame->local_cuboids[mpLastKeyFrame->keypoint_associate_objectID[pixelindLastKf]];
                        g2o::cuboid cube_pose_lastkf;
                        if (objectLastframe->already_associated)
                            cube_pose_lastkf = objectLastframe->associated_landmark->allDynamicPoses[mpLastKeyFrame].first;
                        else
                            cube_pose_lastkf = objectLastframe->GetWorldPos();
                        // get new cube pose in this frame??? based on keypoint object asscoiate id.
                        MapObject *objectThisframe;
                        if (use_dynamic_klt_features)
                            objectThisframe = mpCurrentKeyFrame->local_cuboids[mCurrentFrame.keypoint_associate_objectID_harris[i]];
                        else
                            objectThisframe = mpCurrentKeyFrame->local_cuboids[mpCurrentKeyFrame->keypoint_associate_objectID[i]];
                        g2o::cuboid cube_pose_now = objectThisframe->GetWorldPos(); //current obj pose, not BA optimimized
                        // check truth tracklet id.
                        if (use_truth_trackid)
                            if (objectLastframe->truth_tracklet_id != objectThisframe->truth_tracklet_id)
                            {
                                std::cout << "Different object tracklet id, possibly due to wrong KLT point tracking" << std::endl;
                                continue;
                            }
                        g2o::SE3Quat objecttransform = cube_pose_now.pose * cube_pose_lastkf.pose.inverse();
                        cv::Mat Tcw_now_withdynamic = Tcw_now * Converter::toCvMat(objecttransform);

                        cv::KeyPoint kp1, kp2;
                        if (use_dynamic_klt_features)
                        {
                            kp1 = mpLastKeyFrame->mvKeysHarris[pixelindLastKf];
                            kp2 = mCurrentFrame.mvKeysHarris[i];
                        }
                        else
                        {
                            kp1 = mpLastKeyFrame->mvKeysUn[pixelindLastKf];
                            kp2 = mpCurrentKeyFrame->mvKeysUn[i];
                        }
                        // Check parallax between rays
                        cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.0);
                        cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx1) * invfx1, (kp2.pt.y - cy1) * invfy1, 1.0);

                        cv::Mat x3D;
                        {
                            // Linear Triangulation Method
                            cv::Mat A(4, 4, CV_32F);
                            A.row(0) = xn1.at<float>(0) * Tcw_last.row(2) - Tcw_last.row(0);
                            A.row(1) = xn1.at<float>(1) * Tcw_last.row(2) - Tcw_last.row(1);
                            A.row(2) = xn2.at<float>(0) * Tcw_now_withdynamic.row(2) - Tcw_now_withdynamic.row(0);
                            A.row(3) = xn2.at<float>(1) * Tcw_now_withdynamic.row(2) - Tcw_now_withdynamic.row(1);

                            cv::Mat w, u, vt;
                            cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                            x3D = vt.row(3).t();

                            if (x3D.at<float>(3) == 0)
                                continue;

                            // Euclidean coordinates
                            x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
                        }

                        if ((Converter::toVector3f(x3D) - pMP->GetWorldPosVec()).norm() > 5) // if too far from object center, not good triangulation.
                        {
                            continue;
                        }
                        pMP->is_triangulated = true;
                        pMP->PosToObj = Converter::toCvMat(cube_pose_now.pose.inverse().map(Converter::toVector3d(x3D)));
                    }
                }
            }
        }

        if (1) // randomly select N points, don't initialize all of them
        {
            bool actually_use_obj_depth = false;
            if (mono_allframe_Obj_depth_init && whether_detect_object && associate_point_with_object)
                if (mpCurrentKeyFrame->keypoint_associate_objectID.size() > 0)
                    actually_use_obj_depth = true;

            if (actually_use_obj_depth)
            {
                cout << "Tracking just about to initialize object depth point" << endl;

                // detect new KLF feature points   away from existing featute points.
                if (use_dynamic_klt_features)
                {
                    // loop over all mappoints, create circle mask
                    //             objmask_img:  0 background, >0 object areas   change to--->   255 object areas. 0 background.
                    int MIN_DIST = 10;                                                            // or 10  20
                    cv::Mat good_mask;                                                            //area to generate new features.
                    threshold(mCurrentFrame.objmask_img, good_mask, 0, 255, cv::THRESH_BINARY); // threshold to be 0,255

                    // final existing object mappoints.
                    vector<pair<int, cv::Point2f>> exist_obj_mappts; //point, obverse time

                    for (size_t i = 0; i < mCurrentFrame.mvpMapPointsHarris.size(); i++)
                    {
                        MapPoint *pMP = mCurrentFrame.mvpMapPointsHarris[i]; //TODO later should be separate mappoint for dynamic
                        if (pMP && pMP->is_dynamic)
                        {
                            exist_obj_mappts.push_back(make_pair(pMP->Observations(), mCurrentFrame.mvKeysHarris[i].pt));
                        }
                    }
                    // sort(exist_obj_mappts.begin(), exist_obj_mappts.end(), [](MapPoint *a, MapPoint *b) { return a->Observations() > b->Observations(); });
                    sort(exist_obj_mappts.begin(), exist_obj_mappts.end(), [](const pair<int, cv::Point2f> &a, const pair<int, cv::Point2f> &b) { return a.first > b.first; });

                    for (auto &it : exist_obj_mappts)
                    {
                        if (good_mask.at<uchar>(it.second) == 255)
                        {
                            cv::circle(good_mask, it.second, MIN_DIST, 0, -1);
                        }
                    }
                    cout << "mCurrentFrame.mvpMapPointsHarris size " << mCurrentFrame.mvpMapPointsHarris.size() << " " << exist_obj_mappts.size() << endl;

                    int max_new_pts = 200; //100
                    vector<cv::Point2f> corners;
                    cv::goodFeaturesToTrack(mpCurrentKeyFrame->mImGray, corners, max_new_pts, 0.1, MIN_DIST, good_mask);

                    int numTracked = mCurrentFrame.mvKeysHarris.size();
                    int numNewfeat = corners.size();
                    int totalfeat = numTracked + numNewfeat;

                    mpCurrentKeyFrame->mvKeysHarris = mCurrentFrame.mvKeysHarris;
                    mpCurrentKeyFrame->mvKeysHarris.resize(totalfeat);
                    mpCurrentKeyFrame->mvpMapPointsHarris = mCurrentFrame.mvpMapPointsHarris;
                    mpCurrentKeyFrame->mvpMapPointsHarris.resize(totalfeat);
                    mpCurrentKeyFrame->keypoint_associate_objectID_harris = mCurrentFrame.keypoint_associate_objectID_harris;
                    mpCurrentKeyFrame->keypoint_associate_objectID_harris.resize(totalfeat);

                    //create and append new detected features.
                    for (int new_fea_ind = 0; new_fea_ind < numNewfeat; new_fea_ind++)
                    {
                        int maskval = int(mCurrentFrame.objmask_img.at<uchar>(corners[new_fea_ind])); //0 background, >0 object id
                        int pixelcubeid = maskval - 1;
                        if (maskval == 0)
                        {
                            std::cout << "Get invalid pixel object index" << std::endl;
                            exit(0);
                        }
                        cv::KeyPoint keypt;
                        keypt.pt = corners[new_fea_ind];
                        mpCurrentKeyFrame->mvKeysHarris[new_fea_ind + numTracked] = keypt;
                        mpCurrentKeyFrame->keypoint_associate_objectID_harris[new_fea_ind + numTracked] = pixelcubeid;

                        float point_depth = mpCurrentKeyFrame->local_cuboids[pixelcubeid]->cube_meas.translation()[2]; // camera z
                        cv::Mat x3D = mpCurrentKeyFrame->UnprojectPixelDepth(corners[new_fea_ind], point_depth);

                        MapPoint *pNewMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);
                        pNewMP->is_dynamic = true;
                        mpCurrentKeyFrame->SetupSimpleMapPoints(pNewMP, new_fea_ind + numTracked); // add to frame observation, add to map.
                        //also changed mvpMapPointsHarris
                        pNewMP->is_triangulated = false;
                        pNewMP->SetWorldPos(x3D); // compute dynamic point to object pose
                    }
                    cout << "tracked/new_created features " << numTracked << " " << numNewfeat << endl;

                    //update total features in mCurrentFrame
                    mCurrentFrame.mvKeysHarris = mpCurrentKeyFrame->mvKeysHarris;
                    mCurrentFrame.mvpMapPointsHarris = mpCurrentKeyFrame->mvpMapPointsHarris;
                    mCurrentFrame.keypoint_associate_objectID_harris = mpCurrentKeyFrame->keypoint_associate_objectID_harris;
                }
                else
                {
                    std::vector<int> has_object_depth_pixel_inds; // points with no depth yet but with matching object and plane

                    bool gridsHasMappt[FRAME_GRID_COLS][FRAME_GRID_ROWS];
                    for (int i = 0; i < FRAME_GRID_COLS; i++)
                        for (int j = 0; j < FRAME_GRID_ROWS; j++)
                            gridsHasMappt[i][j] = false;
                    for (size_t i = 0; i < vpMapPointMatches.size(); i++)
                    {
                        MapPoint *pMP = vpMapPointMatches[i];
                        if (!pMP) //no map point yet. not associated yet
                        {
                            int gridx, gridy;
                            if (mpCurrentKeyFrame->PosInGrid(mpCurrentKeyFrame->mvKeys[i], gridx, gridy))
                            {
                                if (gridsHasMappt[gridx][gridy])
                                    continue;
                            }
                            else
                                continue;

                            if (mpCurrentKeyFrame->keypoint_associate_objectID[i] > -1) // have associated object
                                if (mpCurrentKeyFrame->mvKeys[i].octave < 3)            //HACK for KLT tracking, better just use first octave
                                {
                                    has_object_depth_pixel_inds.push_back(i);
                                    gridsHasMappt[gridx][gridy] = true;
                                }
                        }
                        else
                            raw_depth_pts++;
                    }
                    bool whether_actually_planeobj_init_pt = false;

                    double depth_point_ration_now = raw_depth_pts / total_feature_pts;
                    int max_initialize_pts = 0;
                    if (depth_point_ration_now < 0.30) //0.3
                        whether_actually_planeobj_init_pt = true;
                    max_initialize_pts = std::min(int(total_feature_pts * 0.30) - int(raw_depth_pts), int(has_object_depth_pixel_inds.size()));
                    max_initialize_pts = std::min(max_initialize_pts, 80);

                    cout << "all points to initilaze " << has_object_depth_pixel_inds.size() << " initialized " << max_initialize_pts << endl;
                    int nPoints = 0;

                    if (whether_actually_planeobj_init_pt)
                    {
                        srand(time(NULL));
                        //             random_shuffle ( has_object_depth_pixel_inds.begin(), has_object_depth_pixel_inds.end() );
                        random_unique2(has_object_depth_pixel_inds.begin(), has_object_depth_pixel_inds.end(), max_initialize_pts);

                        int vector_counter = 0;
                        while ((nPoints < max_initialize_pts) && (vector_counter < (int)has_object_depth_pixel_inds.size()))
                        {
                            int pixel_ind = has_object_depth_pixel_inds[vector_counter];
                            float point_depth = -1;
                            cv::Mat x3D;

                            if ((point_depth < 0))
                            {
                                if (mpCurrentKeyFrame->keypoint_associate_objectID[pixel_ind] > -1)
                                {
                                    point_depth = mpCurrentKeyFrame->local_cuboids[mpCurrentKeyFrame->keypoint_associate_objectID[pixel_ind]]->cube_meas.translation()[2]; // camera z
                                    x3D = mpCurrentKeyFrame->UnprojectDepth(pixel_ind, point_depth);
                                }
                            }
                            if (point_depth > 0)
                            {
                                MapPoint *pNewMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);
                                mpCurrentKeyFrame->SetupSimpleMapPoints(pNewMP, pixel_ind); // add to frame observation, add to map.
                                pNewMP->is_triangulated = false;
                                nPoints++;
                                if (whether_dynamic_object)
                                {
                                    pNewMP->is_dynamic = true;
                                }
                            }
                            else
                            {
                                //NOTE projected point is negative. remove association? because this point is bad
                            }
                            vector_counter++;
                        }
                        plane_object_initialized_pts = nPoints;
                        std::cout << "Online depth create mappoints!!!!!!!! " << nPoints << std::endl;
                    }
                }
            }
        }

        //     std::cout<<"Finish create my map points!!!!"<<std::endl;
        mCurrentFrame.mvpMapPoints = mpCurrentKeyFrame->GetMapPointMatches();
        // mCurrentFrame.mvpMapPoints indepent of new created mappoints in localmapping
    }

    if (enable_ground_height_scale)
    {
        float img_width = float(mpMap->img_width);
        float img_height = float(mpMap->img_height);

        // do it in every frame, otherwise may take longer time when do it together for many frames.
        for (size_t iMP = 0; iMP < mCurrentFrame.mvpMapPoints.size(); iMP++)
            if (pKF->mvKeysUn[iMP].pt.x > img_width / ground_roi_middle && pKF->mvKeysUn[iMP].pt.x < img_width / ground_roi_middle * (ground_roi_middle - 1))
                if (pKF->mvKeysUn[iMP].pt.y > img_height / ground_roi_lower * (ground_roi_lower - 1)) // lower 1/3, I checked kitti sequence, roughly true.
                {
                    bool not_in_object = true;
                    if (pKF->keypoint_inany_object.size() > 0)
                        if (pKF->keypoint_inany_object[iMP])
                            not_in_object = false;
                    if (not_in_object)
                        pKF->ground_region_potential_pts.push_back(iMP); // used for latter adjacent frame ground fitting
                }

        if (pKF->mnId % ground_everyKFs == 0)
        {
            unsigned long anchor_frame_kfid = 0;
            if (int(pKF->mnId) > ground_everyKFs)
                anchor_frame_kfid = pKF->mnId - ground_everyKFs;
            KeyFrame *first_keyframe = nullptr;
            std::vector<KeyFrame *> ground_local_KFs;
            unsigned long minKFid = pKF->mnId;
            for (size_t ii = 0; ii < mvpLocalKeyFrames.size(); ii++)
            {
                KeyFrame *pKFi = mvpLocalKeyFrames[ii];
                if (pKFi->mnId >= anchor_frame_kfid)
                {
                    ground_local_KFs.push_back(pKFi);
                    pKFi->mnGroundFittingForKF = pKF->mnId;
                    if (pKFi->mnId < minKFid)
                    {
                        minKFid = pKFi->mnId; // the original anchor frame id might not exist due to culling.
                        first_keyframe = pKFi;
                    }
                }
            }
            if (first_keyframe == nullptr)
            {
                std::cout << "Not found first keyframe!!!" << std::endl;
                exit(0);
            }
            ground_local_KFs.push_back(pKF);
            anchor_frame_kfid = pKF->mnId;
            int initializer_starting_frame_id = (*mlpReferences.begin())->mnFrameId; // a fixed value

            KeyFrame *median_keyframe = nullptr; // scale relative to the center frames instead of the begining frame? more accurate?
            if (height_esti_history.size() > 0)
            {
                vector<unsigned> range_kf_ids;
                for (size_t i = 0; i < ground_local_KFs.size(); i++)
                    range_kf_ids.push_back(ground_local_KFs[i]->mnId);
                sort(range_kf_ids.begin(), range_kf_ids.end());
                unsigned median_frameid = range_kf_ids[range_kf_ids.size() / 2];
                for (size_t i = 0; i < ground_local_KFs.size(); i++)
                    if (ground_local_KFs[i]->mnId == median_frameid)
                    {
                        median_keyframe = ground_local_KFs[i];
                        break;
                    }
                if (median_keyframe == nullptr)
                {
                    std::cout << "Not found median keyframe!!!" << std::endl;
                    exit(0);
                }
            }
            else
                median_keyframe = first_keyframe; // still want to scale at the very first fram

            bool recently_have_object = false;
            if (recently_have_object)
                std::cout << "Found cuboid landmark in this range" << std::endl;
            if ((!recently_have_object) || (height_esti_history.size() < 1))
            {
                pcl::PointCloud<pcl::PointXYZ> cloud;
                pcl::PointXYZ pt;
                cloud.points.reserve(mvpLocalMapPoints.size());
                vector<MapPoint *> potential_plane_points;
                for (size_t i = 0; i < ground_local_KFs.size(); i++)
                {
                    std::vector<MapPoint *> framepointmatches = ground_local_KFs[i]->GetMapPointMatches();
                    for (size_t j = 0; j < ground_local_KFs[i]->ground_region_potential_pts.size(); j++)
                    {
                        MapPoint *pMP = framepointmatches[ground_local_KFs[i]->ground_region_potential_pts[j]];
                        if (pMP)
                            if (!pMP->isBad())
                                if (pMP->mnGroundFittingForKF != pKF->mnId)
                                {
                                    cv::Mat point_position = pMP->GetWorldPos(); // fit plane in global frame, then tranform plane. saving time for point transformation
                                    pt.x = point_position.at<float>(0);
                                    pt.y = point_position.at<float>(1);
                                    pt.z = point_position.at<float>(2);
                                    cloud.points.push_back(pt);
                                    potential_plane_points.push_back(pMP);
                                    pMP->mnGroundFittingForKF = pKF->mnId;
                                }
                    }
                }
                std::cout << "Potential plane pt size " << potential_plane_points.size() << " " << ground_local_KFs.size() << std::endl;

                // TODO can we directly search height plane to find points supporting it?? not using ransac. Song used it.
                pcl::SACSegmentation<pcl::PointXYZ> *seg = new pcl::SACSegmentation<pcl::PointXYZ>();
                seg->setOptimizeCoefficients(true);
                seg->setModelType(pcl::SACMODEL_PLANE);
                seg->setMethodType(pcl::SAC_RANSAC);
                if (height_esti_history.size() > 0)
                    seg->setDistanceThreshold(ground_dist_ratio * height_esti_history.back());
                else
                    seg->setDistanceThreshold(0.005); // the raw map is scaled to mean 1.
                pcl::ModelCoefficients coefficients;
                pcl::PointIndices inliers;
                seg->setInputCloud(cloud.makeShared());
                seg->segment(inliers, coefficients);

                Eigen::Vector4f global_fitted_plane(coefficients.values[0], coefficients.values[1], coefficients.values[2], coefficients.values[3]);
                float cam_plane_dist, angle_diff_normal;

                // transform to anchor frame
                KeyFrame *anchor_frame = first_keyframe; // first_keyframe  median_keyframe  pKF;
                cv::Mat anchor_Tcw = anchor_frame->GetPose();
                cv::Mat anchor_Twc = anchor_frame->GetPoseInverse();

                // take averge of all camera pose dist to plane,  not just wrt anchor frame
                if (1)
                {
                    float sum_cam_dist = 0;
                    float sum_angle_diff = 0;
                    vector<float> temp_dists;
                    for (size_t i = 0; i < ground_local_KFs.size(); i++)
                    {
                        KeyFrame *localkf = ground_local_KFs[i];
                        cv::Mat cam_Twc = localkf->GetPoseInverse();
                        Eigen::Matrix4f cam_Twc_eig = Converter::toMatrix4f(cam_Twc);
                        Eigen::Vector4f local_kf_plane = cam_Twc_eig.transpose() * global_fitted_plane;
                        local_kf_plane = local_kf_plane / local_kf_plane.head<3>().norm(); // normalize the plane.

                        float local_cam_plane_dist = fabs(local_kf_plane(3));
                        float local_angle_diff_normal = acos(local_kf_plane.head(3).dot(Vector3f(0, 1, 0))) * 180.0 / M_PI; // 0~pi
                        if (local_angle_diff_normal > 90)
                            local_angle_diff_normal = 180.0 - local_angle_diff_normal;
                        sum_cam_dist += local_cam_plane_dist;
                        sum_angle_diff += local_angle_diff_normal;
                        temp_dists.push_back(local_cam_plane_dist);
                    }
                    cam_plane_dist = sum_cam_dist / float(ground_local_KFs.size());
                    angle_diff_normal = sum_angle_diff / float(ground_local_KFs.size());
                }

                std::cout << "Find init plane dist " << cam_plane_dist << " angle " << angle_diff_normal << " inliers " << inliers.indices.size() << std::endl;

                if (int(inliers.indices.size()) > ground_inlier_pts) // or ratio
                {
                    if (angle_diff_normal < 10)
                    {
                        // for kitti 02, unstale initialization. needs more times
                        if ((fabs(cam_plane_dist - nominal_ground_height) < 0.6) || (height_esti_history.size() < 4)) // or compare with last time?
                        {
                            height_esti_history.push_back(cam_plane_dist);

                            if (height_esti_history.size() == 1)
                            {
                                first_absolute_scale_frameid = first_keyframe->mnFrameId;
                                first_absolute_scale_framestamp = first_keyframe->mTimeStamp;
                            }

                            for (size_t i = 0; i < inliers.indices.size(); i++)
                                potential_plane_points[inliers.indices[i]]->ground_fitted_point = true;

                            float final_filter_height = cam_plane_dist;
                            // take average or recent tow/three frames. or median filter? is this correct if object scale???
                            if (height_esti_history.size() > 2)
                            {
                                final_filter_height = 0.6 * height_esti_history.back() + 0.4 * filtered_ground_height;
                            }
                            filtered_ground_height = final_filter_height;

                            float scaling_ratio = nominal_ground_height / final_filter_height;
                            if (height_esti_history.size() > 1) // ignore the first time.
                            {
                                // don't want too large scaling, which might be wrong...
                                scaling_ratio = std::min(std::max(scaling_ratio, 0.7f), 1.3f);
                            }
                            std::cout << "Actually scale map and frames~~~~~~~~~~~~~~~~" << std::endl;

                            if (enable_ground_height_scale)
                            {
                                for (size_t iMP = 0; iMP < mvpLocalMapPoints.size(); iMP++) // approximatedly. actually mvpLocalMapPoints has much more points
                                {
                                    cv::Mat local_pt = anchor_Tcw.rowRange(0, 3).colRange(0, 3) * mvpLocalMapPoints[iMP]->GetWorldPos() + anchor_Tcw.rowRange(0, 3).col(3);
                                    cv::Mat scaled_global_pt = anchor_Twc.rowRange(0, 3).colRange(0, 3) * (local_pt * scaling_ratio) + anchor_Twc.rowRange(0, 3).col(3);
                                    mvpLocalMapPoints[iMP]->SetWorldPos(scaled_global_pt);
                                }
                                for (size_t iKF = 0; iKF < ground_local_KFs.size(); iKF++)
                                {
                                    cv::Mat anchor_to_pose = ground_local_KFs[iKF]->GetPose() * anchor_Twc;
                                    anchor_to_pose.col(3).rowRange(0, 3) = anchor_to_pose.col(3).rowRange(0, 3) * scaling_ratio;
                                    ground_local_KFs[iKF]->SetPose(anchor_to_pose * anchor_Tcw);
                                }
                                cv::Mat anchor_to_pose = mLastFrame.mTcw * anchor_Twc;
                                anchor_to_pose.col(3).rowRange(0, 3) = anchor_to_pose.col(3).rowRange(0, 3) * scaling_ratio;
                                mLastFrame.SetPose(anchor_to_pose * anchor_Tcw);
                                mCurrentFrame.SetPose(pKF->GetPose());
                                mVelocity.col(3).rowRange(0, 3) = mVelocity.col(3).rowRange(0, 3) * scaling_ratio;

                                // loop over mlpReferences, if any frames' references frames lie in this range, scale the relative poses accordingly
                                // mlpReferences doesn't include the initialization stage... // if it is bad...??
                                for (size_t ind = first_keyframe->mnFrameId - initializer_starting_frame_id; ind < mlpReferences.size(); ind++)
                                    if (mlpReferences[ind]->mnGroundFittingForKF == pKF->mnId)
                                    {
                                        cv::Mat Tcr = mlRelativeFramePoses[ind]; // reference to current
                                        Tcr.col(3).rowRange(0, 3) = Tcr.col(3).rowRange(0, 3) * scaling_ratio;
                                        mlRelativeFramePoses[ind] = Tcr;
                                    }
                            }
                        }
                        else
                            std::cout << "Too large change compared to last time. " << cam_plane_dist << " last " << filtered_ground_height << std::endl;
                    }
                    else
                        std::cout << "Bad ground orientation." << std::endl;
                }
                else
                    std::cout << "Not enough inliers." << std::endl;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF); // call local mapping to insert the kew KF,  map will add it.

    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP)
        {
            if (pMP->isBad())
            {
                *vit = static_cast<MapPoint *>(NULL);
            }
            else
            {
                if (pMP->is_dynamic)
                    continue;
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end(); vit != vend; vit++)
    {
        MapPoint *pMP = *vit;
        if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if (pMP->isBad())
            continue;
        if (pMP->is_dynamic)
            continue;

        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5)) // project point.
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8); //TODO  if point is from my depth init, whether increase the search radius????
        int searchRadiusFactor = 1;
        if (mSensor == System::RGBD)
            searchRadiusFactor = 3;
        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
            searchRadiusFactor = 5;
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, searchRadiusFactor);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        KeyFrame *pKF = *itKF;
        const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

        for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end(); itMP != itEndMP; itMP++)
        {
            MapPoint *pMP = *itMP;
            if (!pMP)
                continue;
            if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                continue;
            if (!pMP->isBad())
            {
                if (whether_dynamic_object && pMP->is_dynamic)
                    continue;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame *, int> keyframeCounter;
    for (int i = 0; i < mCurrentFrame.N; i++)
    {
        if (mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
            if (!pMP->isBad())
            {
                if (whether_dynamic_object && pMP->is_dynamic)
                    continue;
                const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end(); it != itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i] = NULL;
            }
        }
    }

    if (keyframeCounter.empty())
        return;

    int max = 0;
    KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end(); it != itEnd; it++)
    {
        KeyFrame *pKF = it->first;

        if (pKF->isBad())
            continue;

        if (it->second > max) // find keyframe shares most points
        {
            max = it->second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end(); itKF != itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if (mvpLocalKeyFrames.size() > 80)
            break;

        KeyFrame *pKF = *itKF;

        const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end(); itNeighKF != itEndNeighKF; itNeighKF++)
        {
            KeyFrame *pNeighKF = *itNeighKF;
            if (!pNeighKF->isBad())
            {
                if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame *> spChilds = pKF->GetChilds();
        for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++)
        {
            KeyFrame *pChildKF = *sit;
            if (!pChildKF->isBad())
            {
                if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame *pParent = pKF->GetParent();
        if (pParent)
        {
            if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                break;
            }
        }
    }

    if (pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if (vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75, true);

    vector<PnPsolver *> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint *>> vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (int i = 0; i < nKFs; i++)
    {
        KeyFrame *pKF = vpCandidateKFs[i];
        if (pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
            if (nmatches < 15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (int i = 0; i < nKFs; i++)
        {
            if (vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver *pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint *> sFound;

                const int np = vbInliers.size();

                for (int j = 0; j < np; j++)
                {
                    if (vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j] = NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.N; io++)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.N; io++)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = NULL;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }
}

void Tracking::Reset()
{
    mpViewer->RequestStop();

    cout << "System Reseting" << endl;
    while (!mpViewer->isStopped())
        usleep(3000);

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if (mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer *>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

} // namespace ORB_SLAM2
