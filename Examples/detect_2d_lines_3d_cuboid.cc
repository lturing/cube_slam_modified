#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <ctime>
#include <stdio.h>
#include <string>

#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

#include "line_lbd_allclass.h"
#include "line_descriptor.hpp"
#include "detect_3d_cuboid.h"
#include "detector_opencv_dnn.h"
#include "segmentor_opencv_dnn.h"
#include "detector_onnxruntime.h"
#include "segmentor_onnxruntime.h"

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace std;

void detect_2d(Detector_ONNXRUNTIME* myDetectorOnnxRun, cv::Mat& bgr_img, MatrixXd& obj_bbox_coors, string& save_folder, std::vector<std::string>& _classNamesList);
void detect_lines(const cv::Mat& bgr_img, string save_folder, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &result_mat);
void detect_cuboids(const cv::Mat& bgr_img, const Matrix3d Kalib, const Eigen::Matrix4d &transToWolrd, const Eigen::MatrixXd &obj_bbox_coors, Eigen::MatrixXd& all_lines_raw, std::vector<ObjectSet> &all_object_cuboids);
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps);

int main( int argc, char** argv )
{
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

    //const std::string  k_detect_model_path = "yolov5-onnxruntime/models/yolov5m.onnx";
    std::shared_ptr<Detector_ONNXRUNTIME> myDetectorOnnxRun = std::make_shared<Detector_ONNXRUNTIME>();
    
    auto batchSize = 1;
    auto inputSize = cv::Size(640, 640);
    std::string modelPath = "/home/spurs/x/yolov8/yolov8s.onnx";
        
    myDetectorOnnxRun->LoadModel(modelPath);
    myDetectorOnnxRun->setClassNames(_classNamesList);
    myDetectorOnnxRun->setBatchSize(batchSize);
    myDetectorOnnxRun->setInputSize(inputSize);

    Matrix3d Kalib;
    Kalib << 721.538, 0, 609.559,
             0, 721.538, 172.854,
             0,       0,       1;

    Matrix4d transToWolrd;
    // hard coded  NOTE if accurate camera roll/pitch, could sample it!
    transToWolrd << 1, 0, 0, 0,
                    0, 0, 1, 0,
                    0, -1, 0, 1.65,
                    0, 0, 0, 1;
    
    /*
    Matrix3d Kalib;
    Kalib << 529.5000, 0, 365.0000,
        0, 529.5000, 265.0000,
        0, 0, 1.0000;

    Matrix4d transToWolrd;
    transToWolrd << 1, 0.0011, 0.0004, 0, // hard coded  NOTE if accurate camera roll/pitch, could sample it!
        0, -0.3376, 0.9413, 0,
        0.0011, -0.9413, -0.3376, 1.35,
        0, 0, 0, 1;
    */

    //std::string image_path = "Examples/data/kitti_2011_10_03_0000000168.png";
    //std::string image_path = "Examples/data/0000_rgb_raw.jpg";

    // Retrieve paths to images
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    string data_folder = "/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02";

    LoadImages(data_folder, vstrImageFilenames, vTimestamps);
    int nImages = vstrImageFilenames.size();

    for (int ni = 0; ni < vstrImageFilenames.size(); ni++)
    {
        //std::string image_path = "/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/data/0000000095.png";
        std::string image_path = vstrImageFilenames[ni];
        cv::Mat bgr_img = imread(image_path, cv::IMREAD_COLOR); // BGR
        if( bgr_img.data == NULL )
        {
            std::cout << "Error, image could not be loaded. Please, check its path "<< image_path << std::endl;
            return -1;
        }
        if( bgr_img.channels() == 1 )
            cvtColor(bgr_img, bgr_img, COLOR_GRAY2BGR );
        
        std::string save_folder = "Examples/data/";
        MatrixXd obj_bbox_coors(1, 5);                // hard coded
        //obj_bbox_coors << 188, 189, 201, 311, 0.8800; // [x y w h prob]
        //obj_bbox_coors.leftCols<2>().array() -= 1;    // change matlab coordinate to c++, minus 1

        if (0)
        {
            cv::Mat imGray;
            cv::cvtColor(bgr_img, imGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(imGray, bgr_img, cv::COLOR_GRAY2BGR);
        }
        detect_2d(myDetectorOnnxRun.get(), bgr_img, obj_bbox_coors, save_folder, _classNamesList);
        //std::cout << "detect " << obj_bbox_coors.rows() << " 2d objects" << std::endl;

        // remove some 2d boxes too close to boundary.
        int boundary_threshold = 10;
        int img_width = bgr_img.cols;
        int img_height = bgr_img.rows;

        std::vector<int> good_object_ids;
        for (int i = 0; i < obj_bbox_coors.rows(); i++)
        {
            if ((obj_bbox_coors(i, 0) > boundary_threshold) && (obj_bbox_coors(i, 0) + obj_bbox_coors(i, 2) < img_width - boundary_threshold) &&
                (obj_bbox_coors(i, 1) > boundary_threshold) && (obj_bbox_coors(i, 1) + obj_bbox_coors(i, 3) < img_height - boundary_threshold))
            {
                good_object_ids.push_back(i);
            }
            //else 
            //    std::cout << "obj_bbox_coors(i)=" << obj_bbox_coors.row(i) << ", img_width=" << img_width << ", img_height=" << img_height << std::endl;
        }
            
        Eigen::MatrixXd all_obj2d_bbox_infov_mat(good_object_ids.size(), 5);
        for (size_t i = 0; i < good_object_ids.size(); i++)
        {
            all_obj2d_bbox_infov_mat.row(i) = obj_bbox_coors.row(good_object_ids[i]);
        }

        if (all_obj2d_bbox_infov_mat.rows() < 1)
        {
            std::cout << "failed to detect 2d object" << std::endl;
            //std::cout << "good_object_ids.size()=" << good_object_ids.size() << std::endl;
            continue;
            //return 1;
        }

        std::vector<ObjectSet> all_object_cuboids;

        Eigen::MatrixXd all_lines_raw(100, 4); // 100 is some large frame number,   the txt edge index start from 0

        detect_lines(bgr_img, save_folder, all_lines_raw);
        if (all_lines_raw.rows() < 1)
        {
            std::cout << "failed to detect lines" << std::endl;
            return 1;
        }

        detect_cuboids(bgr_img, Kalib, transToWolrd, all_obj2d_bbox_infov_mat, all_lines_raw, all_object_cuboids);
    }

    return 0;
}

void detect_2d(Detector_ONNXRUNTIME* myDetectorOnnxRun, cv::Mat& bgr_img, MatrixXd& obj_bbox_coors, string& save_folder, std::vector<std::string>& _classNamesList)
{
    cv::Mat frame = bgr_img.clone();
    bool save_to_imgs = false; 
    bool print_detect = false;

    std::vector<cv::Mat> imgBatch;
    //imgBatch.clear();
    imgBatch.push_back(frame);

    //Second/Millisecond/Microsecond  秒s/毫秒ms/微秒us
    auto start = std::chrono::system_clock::now();
    auto results = myDetectorOnnxRun->Run(imgBatch);
    auto end = std::chrono::system_clock::now();
    auto detect_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();//ms
    //std::cout << "cost " << detect_time / 1000.0 << " seconds" << std::endl;
    
    if (save_to_imgs)
    {
        auto color_box = cv::Scalar(0, 0, 255);
        for (int i = 0; i < results[0].size(); ++i) {
            cv::rectangle(frame, results[0][i].box, color_box, 2, 8);
            cv::putText(frame, _classNamesList[results[0][i].classID],
                                cv::Point(results[0][i].box.x, results[0][i].box.y),
                                cv::FONT_HERSHEY_COMPLEX, 0.5, CV_RGB(0,255,0), 1.0);
        }
        cv::imshow("YOLOV8 Detection", frame);
        cv::waitKey();
        std::string output_file = save_folder + "2dObject.png";
        cv::imwrite(output_file, frame);
    }
    
    int row_counter = 0;
    for(int i = 0; i < results[0].size(); i++)
    {
        // [x y w h prob]
        obj_bbox_coors(i, 0) = results[0][i].box.tl().x;
        obj_bbox_coors(i, 1) = results[0][i].box.tl().y;
        obj_bbox_coors(i, 2) = results[0][i].box.size().width; //w;
        obj_bbox_coors(i, 3) = results[0][i].box.size().height; // h;
        obj_bbox_coors(i, 4) = results[0][i].confidence;
        row_counter += 1;
        if (row_counter >= obj_bbox_coors.rows()) // if matrix row is not enough, make more space.
            obj_bbox_coors.conservativeResize(obj_bbox_coors.rows() * 2, obj_bbox_coors.cols());
    }
    results.clear();
    obj_bbox_coors.conservativeResize(row_counter, obj_bbox_coors.cols()); // cut into actual rows

    if (print_detect)
    {
        std::cout << "detect " << obj_bbox_coors.rows() << " 2d objects" << std::endl;
        for (int i = 0; i < obj_bbox_coors.rows(); i++)
        {
            std::cout << "x y w h=" << obj_bbox_coors(i, 0) << " " << obj_bbox_coors(i, 1) << " " << obj_bbox_coors(i, 2) << " " << obj_bbox_coors(i, 3) << std::endl;
        }
    }
}

void detect_lines(const cv::Mat& bgr_img, string save_folder, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &result_mat)
{
    bool use_LSD_algorithm = true;
    bool save_to_imgs = false;
    bool save_to_txts = false;
    bool plot_line = false;

    int numOfOctave_ = 1;
    float Octave_ratio = 2.0;  

    line_lbd_detect* line_lbd_ptr = new line_lbd_detect(numOfOctave_,Octave_ratio); 
    line_lbd_ptr->use_LSD = use_LSD_algorithm;
    line_lbd_ptr->line_length_thres = 15;  // remove short edges

    // using my line detector class, could select LSD or edline.
    cv::Mat out_edges;
    std::vector< KeyLine> keylines_raw,keylines_out;
    line_lbd_ptr->detect_raw_lines(bgr_img,keylines_raw);
    line_lbd_ptr->filter_lines(keylines_raw,keylines_out);  // remove short lines

    cv::Mat bgr_img_cp;
    drawKeylines(bgr_img, keylines_out, bgr_img_cp, cv::Scalar( 0, 150, 0 ),2); // B G R
    
    if (plot_line)
    {
        imshow( "Line detector", bgr_img_cp );
        waitKey();
    }

    if (save_to_imgs)
    {
        std::string img_save_name = save_folder+"saved_edges.jpg";
        cv::imwrite(img_save_name, bgr_img_cp);
    }
      
    if (save_to_txts)
    {
        std::string txt_save_name = save_folder+"saved_edges.txt";
        ofstream resultsFile;
        resultsFile.open(txt_save_name);
        for (int j=0;j<keylines_out.size();j++)
        {
            resultsFile <<keylines_out[j].startPointX <<"\t" <<keylines_out[j].startPointY  <<"\t"
                <<keylines_out[j].endPointX   <<"\t" <<keylines_out[j].endPointY    <<endl;
        }
        resultsFile.close();
    }

    int row_counter = 0;
    for (int i = 0; i < keylines_out.size(); i++)
    {
        result_mat(i, 0) = keylines_out[i].startPointX;
        result_mat(i, 1) = keylines_out[i].startPointY;
        result_mat(i, 2) = keylines_out[i].endPointX;
        result_mat(i, 3) = keylines_out[i].endPointY;

        row_counter += 1;
        if (row_counter >= result_mat.rows()) // if matrix row is not enough, make more space.
            result_mat.conservativeResize(result_mat.rows() * 2, result_mat.cols());
    }

    result_mat.conservativeResize(row_counter, result_mat.cols()); // cut into actual rows
}

void detect_cuboids(const cv::Mat& bgr_img, const Matrix3d Kalib, const Eigen::Matrix4d &transToWolrd, const Eigen::MatrixXd &obj_bbox_coors, Eigen::MatrixXd& all_lines_raw, std::vector<ObjectSet> &all_object_cuboids)
{
    detect_3d_cuboid detect_cuboid_obj;
    detect_cuboid_obj.whether_plot_detail_images = false;
    detect_cuboid_obj.whether_plot_final_images = true;
    detect_cuboid_obj.print_details = false; // false  true
    detect_cuboid_obj.set_calibration(Kalib);
    detect_cuboid_obj.whether_sample_bbox_height = false;
    detect_cuboid_obj.whether_sample_cam_roll_pitch = true;

    detect_cuboid_obj.detect_cuboid(bgr_img, transToWolrd, obj_bbox_coors, all_lines_raw, all_object_cuboids);

}


void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/data/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(10) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}
