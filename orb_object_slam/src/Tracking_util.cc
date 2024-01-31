
#include "Tracking.h"

#include "Map.h"
#include "MapPoint.h"
#include "MapObject.h"
#include "Converter.h"
#include "KeyFrame.h"

// by me
#include "Parameters.h"
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <boost/filesystem.hpp>

#include <typeinfo>
#include <glob.h>

using namespace std;
using namespace Eigen;

namespace ORB_SLAM2
{

std::string Tracking::GetFileName(std::string filePath, bool withExtension)
{
    // Create a Path object from File Path
    boost::filesystem::path pathObj(filePath);
    // Check if file name is required without extension
    if(withExtension == false)
    {   
        // Check if file has stem i.e. filename without extension
        if(pathObj.has_stem())
        {
            // return the stem (file name without extension) from path object
            return pathObj.stem().string();
        }
        return ""; 
    }   
    else
    {   
        // return the file name with extension from path object
        return pathObj.filename().string();
    }   
}


void Tracking::ReadAllObjecttxt()
{
    int total_img_ind = 10000;                // some large number
    //all_offline_object_cubes.reserve(2000); // vector will double capacity automatically after full
    bool set_all_obj_probto_one = false;

    if (use_truth_trackid)
        std::cout << "Read ground truth object tracking id" << std::endl;

    std::vector<string> pred_frame_obj_txts;
    glob_t glob_result;
    glob((base_data_folder + "/" + cuboid_folder + "/*txt").c_str(), GLOB_TILDE, NULL, &glob_result);

    for(unsigned int i=0; i<glob_result.gl_pathc; ++i)
    {   
        std::stringstream ss(glob_result.gl_pathv[i]);
        string name;
        ss >> name;
        pred_frame_obj_txts.push_back(name);
    }  
    
    std::sort(pred_frame_obj_txts.begin(), pred_frame_obj_txts.end()); // ascend order
    std::cout << "read " << pred_frame_obj_txts.size() << " cuboid files" << std::endl;
    for (int i = 0; i < pred_frame_obj_txts.size(); i++)
    {
        std::string pred_frame_obj_txt = pred_frame_obj_txts[i];
        std::string name = GetFileName(pred_frame_obj_txt, false);
        //3d cuboid txts:  each row:  [cuboid_center(3), yaw, cuboid_scale(3), [x1 y1 w h]], prob
        int data_width = 12;
        if (use_truth_trackid)
            data_width = 13;
        Eigen::MatrixXd pred_frame_objects(5, data_width);
        if (read_all_number_txt(pred_frame_obj_txt, pred_frame_objects))
        {
            if (set_all_obj_probto_one)
                for (int ii = 0; ii < pred_frame_objects.rows(); ii++)
                    pred_frame_objects(ii, data_width - 1) = 1;

            //all_offline_object_cubes.push_back(pred_frame_objects);
            all_offline_object_cubes[name] = pred_frame_objects;
            if (!use_truth_trackid)
                if (pred_frame_objects.rows() > 0)
                    for (int ii = 0; ii < pred_frame_objects.rows(); ii++)
                        if (pred_frame_objects(ii, data_width - 1) < -0.1)
                            std::cout << "Read offline Bad object prob " << pred_frame_objects(ii, data_width - 1) << "   from file  " << pred_frame_obj_txt << "  row  " << ii << std::endl;
        }
        else
        {
            std::cout << "failed to read " << pred_frame_obj_txt << std::endl;
            break;
        }
    }
}

void Tracking::SaveOptimizedCuboidsToTxt()
{
    std::cout << "Save optimized cuboids into txts!!!" << std::endl;

    g2o::SE3Quat InitToGround_se3 = Converter::toSE3Quat(InitToGround);

    //directly record global object pose, into one txt
    vector<MapObject *> all_Map_objs = mpMap->GetAllMapObjects();
    std::string save_object_pose_txt = base_data_folder + "/slam_output/orb_opti_pred_objs.txt";
    std::string save_object_pose_temp_txt = base_data_folder + "/slam_output/orb_opti_pred_objs_temp.txt";
    if (boost::filesystem::exists(save_object_pose_txt))
    {
        std::cout << "Rename file base_data_folder/slam_output/orb_opti_pred_objs" << std::endl;
        if (boost::filesystem::exists(save_object_pose_temp_txt))
            boost::filesystem::remove(save_object_pose_temp_txt);
        boost::filesystem::rename(save_object_pose_txt, save_object_pose_temp_txt); // delete existing files as keyframes are different each run.
    }

    if (!whether_dynamic_object) // for static object, record final pose
    {
        int obj_counter = 0;
        save_final_optimized_cuboids.open(save_object_pose_txt.c_str());
        for (size_t i = 0; i < all_Map_objs.size(); i++)
        {
            MapObject *pMO = all_Map_objs[i];
            if (!pMO->isBad())
            {
                pMO->record_txtrow_id = obj_counter++;
                // transform to ground frame which is more visible.
                g2o::cuboid cube_pose_to_init = pMO->GetWorldPos();
                g2o::cuboid cube_pose_to_ground = cube_pose_to_init.transform_from(InitToGround_se3); // absolute ground frame.
                if (build_worldframe_on_ground)
                    cube_pose_to_ground = cube_pose_to_init;
                save_final_optimized_cuboids << pMO->mnId << "  " << pMO->isGood << "  " << cube_pose_to_ground.toVector().transpose() << " "
                                             << "\n";
            }
        }
        save_final_optimized_cuboids.close();
        save_online_detected_cuboids.close();
    }

    if (whether_dynamic_object)
    {
        std::string save_object_velocity_txt = base_data_folder + "/slam_output/orb_opti_pred_objs_velocity.txt";
        std::ofstream Logfile;
        Logfile.open(save_object_velocity_txt.c_str());
        std::cout << "save total object size " << all_Map_objs.size() << std::endl;
        for (size_t i = 0; i < all_Map_objs.size(); i++)
        {
            MapObject *pMO = all_Map_objs[i];
            {
                for (map<KeyFrame *, Eigen::Vector2d, cmpKeyframe>::iterator mit = pMO->velocityhistory.begin(); mit != pMO->velocityhistory.end(); mit++)
                {
                    Logfile << pMO->truth_tracklet_id << "  " << mit->first->mnFrameId << "    " << mit->second.transpose() << "\n";
                }
            }
        }
        Logfile.close();
    }

    // record object pose in each frame, into different txts
    if ((scene_unique_id == kitti) && whether_dynamic_object)
    {
        // for KITTI  for each keyframe, save its observed optimized cuboids into txt.
        const vector<KeyFrame *> all_keyframes = mpMap->GetAllKeyFrames(); // not sequential

        std::string kitti_saved_obj_dir = base_data_folder + "/slam_output/orb_frame_3d/";
        std::string kitti_saved_obj_dir_temp = base_data_folder + "/slam_output/orb_frame_3d_temp/";
        if (whether_dynamic_object)
        {
            kitti_saved_obj_dir = base_data_folder + "/slam_output/orb_obj_3d/";
            kitti_saved_obj_dir_temp = base_data_folder + "/slam_output/orb_obj_3d_temp/";
        }
        if (boost::filesystem::exists(kitti_saved_obj_dir))
        {
            std::cout << "Rename folder base_data_folder/slam_output/orb_frame_3d/" << std::endl;
            if (boost::filesystem::exists(kitti_saved_obj_dir_temp))
                boost::filesystem::remove_all(kitti_saved_obj_dir_temp);
            boost::filesystem::rename(kitti_saved_obj_dir, kitti_saved_obj_dir_temp); // delete existing files as keyframes are different each run.
        }
        boost::filesystem::create_directories(kitti_saved_obj_dir);

        for (size_t i = 0; i < all_keyframes.size(); i++)
        {
            KeyFrame *kf = all_keyframes[i];

            char sequence_frame_index_c[256];
            sprintf(sequence_frame_index_c, "%04d", (int)kf->mnFrameId);
            std::string save_object_ba_pose_txt = kitti_saved_obj_dir + sequence_frame_index_c + "_orb_3d_ba.txt"; // object pose after BA
            ofstream Logfile;
            Logfile.open(save_object_ba_pose_txt.c_str());
            std::string save_object_asso_pose_txt = kitti_saved_obj_dir + sequence_frame_index_c + "_orb_3d_asso.txt"; // object pose before BA, just using association
            ofstream Logfile2;
            Logfile2.open(save_object_asso_pose_txt.c_str());
            g2o::SE3Quat frame_pose_to_init = Converter::toSE3Quat(kf->GetPoseInverse()); // camera to init world

            for (size_t j = 0; j < kf->cuboids_landmark.size(); j++)
            {
                MapObject *pMO = kf->cuboids_landmark[j];

                if (!pMO)
                {
                    continue;
                }

                g2o::cuboid cube_global_pose;
                if (whether_dynamic_object) // get object pose at this frame.
                {
                    if (pMO->allDynamicPoses.count(kf))
                        cube_global_pose = pMO->allDynamicPoses[kf].first;
                    else
                        continue;
                }
                else
                    cube_global_pose = pMO->GetWorldPos();

                g2o::cuboid cube_to_camera = cube_global_pose.transform_to(frame_pose_to_init);        // measurement in local camera frame
                g2o::cuboid cube_to_local_ground = cube_to_camera.transform_from(InitToGround_se3); //some approximation

                int object_ID = pMO->GetIndexInKeyFrame(kf); // obj index in all raw frame cuboids. -1 if bad deleted object
                Logfile << cube_to_local_ground.toMinimalVector().transpose() << "    " << object_ID << "   " << pMO->truth_tracklet_id << "\n";

                if (!whether_dynamic_object) // the cube before optimized!!!
                {
                    cube_to_camera = pMO->pose_noopti.transform_to(frame_pose_to_init);
                    cube_to_local_ground = cube_to_camera.transform_from(InitToGround_se3);
                    Logfile2 << cube_to_local_ground.toMinimalVector().transpose() << "    " << object_ID << "\n";
                }
            }
            Logfile.close();
            Logfile2.close();
        }
    }

    return;
}

} // namespace ORB_SLAM2
