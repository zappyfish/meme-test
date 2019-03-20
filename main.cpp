#include <iostream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace tensorflow;
namespace fs = boost::filesystem;

typedef std::vector<std::pair<std::string, tensorflow::Tensor>> tensor_dict;

cv::Mat loadImageFloat(std::string &path);
Tensor getTensor(cv::Mat &inp1, cv::Mat &inp2, cv::Mat &inp3);
void fillTensor(cv::Mat &src, Tensor &tensor, int start);
cv::Mat getResized(cv::Mat &im);

int main() {
    // set up your input paths
    const string graph_fn = "model-199160.meta";
    const string checkpoint_fn = "model-199160";

    auto options = tensorflow::SessionOptions();
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.2);
    options.config.mutable_gpu_options()->set_allow_growth(true);

    auto session = NewSession(options);
    if (session == nullptr) {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

// Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    TF_CHECK_OK(ReadBinaryProto(tensorflow::Env::Default(), graph_fn, &graph_def));

    TF_CHECK_OK(session->Create(graph_def.graph_def()));


// Read weights from the saved checkpoint
//    Tensor checkpointPathTensor(DT_STRING, TensorShape());
//    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
//
//    status = session->Run(
//            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
//            {},
//            {graph_def.saver_def().restore_op_name()},
//            nullptr);
//    if (!status.ok()) {
//        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
//    }

// and run the inference to your liking

    std::vector<tensorflow::Tensor> outputTensors;

    std::string path = "/home/nvidia/test/images";
    std::vector<std::string> paths;
    for (const auto & entry : fs::directory_iterator(path)) {
        paths.push_back(entry.path().string());
    }
    int num_images = 0;
    cv::Mat imgs[3];

    for (const auto & path: paths) {
        cv::Mat test_img = cv::imread(path);
        imgs[num_images] = getResized(test_img);
        if (num_images < 3) {
            num_images++;
        }
        if (num_images == 3) {
            Tensor imgTensor = getTensor(imgs[0], imgs[1], imgs[2]);
            tensor_dict feedDict = {
                    {"input", imgTensor}
            };
            std::cout << "feed me\n";
            status = session->Run(feedDict, {}, {}, &outputTensors);
            std::cout << "feeded\n";
            imgs[0] = imgs[1];
            imgs[1] = imgs[2]; // shift over
            std::cout << outputTensors.size() << std::endl;
            std::cout << outputTensors[0].tensor<float, 6>() << std::endl;
        }
    }



    return 0;
}

Tensor getTensor(cv::Mat &inp1, cv::Mat &inp2, cv::Mat &inp3) {

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                tensorflow::TensorShape({1, inp1.rows, inp1.cols, 9}));

    fillTensor(inp1, input_tensor, 0);
    fillTensor(inp2, input_tensor, 3);
    fillTensor(inp3, input_tensor, 6);

    return input_tensor;
}

void fillTensor(cv::Mat &src, Tensor &tensor, int start) {
    auto input_tensor_mapped = tensor.tensor<float, 4>();
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(row, col);
            // TODO: check if it should be 0, 1, 2 pixel order instead
            input_tensor_mapped(0, row, col, 0 + start) = pixel[2];
            input_tensor_mapped(0, row, col, 1 + start) = pixel[1];
            input_tensor_mapped(0, row, col, 2 + start) = pixel[0];

        }
    }
}

cv::Mat getResized(cv::Mat &im) {
    cv::Size size(416,128);//the dst image size,e.g.100x100
    cv::Mat dst;//dst image
    cv::resize(src,dst,size);//resize image
    return dst;
}