#include "detector_opencv_dnn.h"
#include "segmentor_opencv_dnn.h"
#include "detector_onnxruntime.h"
#include "segmentor_onnxruntime.h"
#include "BYTETracker.h"
#include <vector>
#include <glob.h>
#include <iostream>
#include <algorithm>



void LoadImages(const std::string &strPathToSequence, std::vector<std::string> &vstrImageFilenames);

int main(int argc, char *argv[])
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

    std::string data_dir = "/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/data";
    std::vector<std::string> vstrImageFilenames;
    LoadImages(data_dir, vstrImageFilenames);

    auto batchSize = 1;
    auto inputSize = cv::Size(640, 640);

    Detector_ONNXRUNTIME* myDetectorOnnxRun = new Detector_ONNXRUNTIME();
    Detector_OpenCV_DNN* myDetectorOnnxCV = new Detector_OpenCV_DNN();

    std::string modelPath = "/home/spurs/x/yolov8/yolov8s.onnx";
    //std::string modelPath = "/home/spurs/x/yolov8/yolov8m.onnx";

    myDetectorOnnxRun->LoadModel(modelPath);
    myDetectorOnnxRun->setClassNames(_classNamesList);
    myDetectorOnnxRun->setBatchSize(batchSize);
    myDetectorOnnxRun->setInputSize(inputSize);

    myDetectorOnnxCV->LoadModel(modelPath);
    myDetectorOnnxCV->setClassNames(_classNamesList);
    myDetectorOnnxCV->setBatchSize(batchSize);
    myDetectorOnnxCV->setInputSize(inputSize);

    //bytetrack
    int fps=10;
    byte_track::BYTETracker bytetracker(fps, 30);

    cv::Mat frame = cv::imread(vstrImageFilenames[0], cv::IMREAD_COLOR);
    cv::VideoWriter video("out.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame.size[1], frame.size[0]));
    auto start = std::chrono::system_clock::now();
    std::vector<cv::Mat> imgBatch;
    for (int imgIDX = 0; imgIDX < vstrImageFilenames.size(); ++imgIDX) {
        // make batch of images = 1
        imgBatch.clear();
        cv::Mat frame = cv::imread(vstrImageFilenames[imgIDX], cv::IMREAD_COLOR);

        cv::Mat imGray;
        if (0)
        {
            cv::cvtColor(frame, imGray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(imGray, frame, cv::COLOR_GRAY2BGR);
        }

        imgBatch.push_back(frame);

        auto result = myDetectorOnnxRun->Run(imgBatch);
        //auto result = myDetectorOnnxCV->Run(imgBatch);

        //std::cout << "detect " << result[0].size() << " objects" << std::endl;
        std::vector<byte_track::STrackPtr> output_stracks = bytetracker.update(result[0]);

        for (unsigned long i = 0; i < output_stracks.size(); i++)
        {
            std::vector<float> tlwh = output_stracks[i]->tlwh;

            // filter 
            bool vertical = tlwh[2] / tlwh[3] > 1.6;
            //if (tlwh[2] * tlwh[3] > 20 && !vertical)
            if (1)
            {
                cv::Scalar s = bytetracker.get_color(output_stracks[i]->track_id);
                cv::putText(frame, std::to_string(output_stracks[i]->track_id) + " " + _classNamesList[output_stracks[i]->classID], cv::Point(tlwh[0], tlwh[1] - 5),
                        cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);

                cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
            }
        }

        cv::imshow("YOLOv8", frame);
        cv::waitKey(1);
        video.write(frame);
        
    }

    auto end = std::chrono::system_clock::now();
    auto detect_time =std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();//ms
    auto mean_time = detect_time / 1000.0 / std::max(1, static_cast<int>(vstrImageFilenames.size()));

    video.release();
    cv::destroyAllWindows();

}


void LoadImages(const std::string &strPathToSequence, std::vector<std::string> &vstrImageFilenames)
{

    glob_t glob_result;
    glob((strPathToSequence + "/*png").c_str(), GLOB_TILDE, NULL, &glob_result);

    for(unsigned int i=0; i<glob_result.gl_pathc; ++i)
    {
        std::stringstream ss(glob_result.gl_pathv[i]);
        std::string name;
        ss >> name;
        vstrImageFilenames.push_back(name);
    }

    std::sort(vstrImageFilenames.begin(), vstrImageFilenames.end()); // ascend order
    //std::cout << vstrImageFilenames[0] << " " << vstrImageFilenames[100] << std::endl;
}
