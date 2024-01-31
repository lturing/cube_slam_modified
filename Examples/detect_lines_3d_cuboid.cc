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

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

#include "line_lbd_allclass.h"
#include "line_descriptor.hpp"
#include "detect_3d_cuboid.h"

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace std;


void detect_lines(const cv::Mat& bgr_img, string save_folder, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &result_mat);
void detect_cuboids(const cv::Mat& bgr_img, const Matrix3d Kalib, const Eigen::Matrix4d &transToWolrd, const Eigen::MatrixXd &obj_bbox_coors, Eigen::MatrixXd& all_lines_raw, std::vector<ObjectSet> &all_object_cuboids);


int main( int argc, char** argv )
{
    /* get parameters from comand line */
    //if(argc<2){
    //  std::cout<<"Provide an image name"<<endl;
    //  return -1;
    //}

    //std::string image_path(argv[1]);

    // hard code, don't change!
    std::string image_path = "Examples/data/0000_rgb_raw.jpg";

    cv::Mat bgr_img = imread(image_path, cv::IMREAD_COLOR); // BGR
    if( bgr_img.data == NULL )
    {
        std::cout << "Error, image could not be loaded. Please, check its path "<<image_path << std::endl;
        return -1;
    }
    if( bgr_img.channels() == 1 )
        cvtColor(bgr_img, bgr_img, COLOR_GRAY2BGR );
    
    Matrix3d Kalib;
    Kalib << 529.5000, 0, 365.0000,
        0, 529.5000, 265.0000,
        0, 0, 1.0000;

    Matrix4d transToWolrd;
    transToWolrd << 1, 0.0011, 0.0004, 0, // hard coded  NOTE if accurate camera roll/pitch, could sample it!
        0, -0.3376, 0.9413, 0,
        0.0011, -0.9413, -0.3376, 1.35,
        0, 0, 0, 1;

    MatrixXd obj_bbox_coors(1, 5);                // hard coded
    obj_bbox_coors << 188, 189, 201, 311, 0.8800; // [x y w h prob]
    obj_bbox_coors.leftCols<2>().array() -= 1;    // change matlab coordinate to c++, minus 1
    std::vector<ObjectSet> all_object_cuboids;

    std::string save_folder = "Examples/data/";
    Eigen::MatrixXd all_lines_raw(100, 4); // 100 is some large frame number,   the txt edge index start from 0

    detect_lines(bgr_img, save_folder, all_lines_raw);
    if (all_lines_raw.rows() < 1)
    {
        std::cout << "failed to detect lines" << std::endl;
        return 1;
    }

    detect_cuboids(bgr_img, Kalib, transToWolrd, obj_bbox_coors, all_lines_raw, all_object_cuboids);

    return 0;
}


void detect_lines(const cv::Mat& bgr_img, string save_folder, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &result_mat)
{
    bool use_LSD_algorithm = true;
    bool save_to_imgs = false;
    bool save_to_txts = false;

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
    imshow( "Line detector", bgr_img_cp );
    waitKey();

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
    detect_cuboid_obj.whether_plot_detail_images = true;
    detect_cuboid_obj.whether_plot_final_images = true;
    detect_cuboid_obj.print_details = false; // false  true
    detect_cuboid_obj.set_calibration(Kalib);
    detect_cuboid_obj.whether_sample_bbox_height = false;
    detect_cuboid_obj.whether_sample_cam_roll_pitch = false;

    detect_cuboid_obj.detect_cuboid(bgr_img, transToWolrd, obj_bbox_coors, all_lines_raw, all_object_cuboids);

}