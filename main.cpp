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
Tensor getTensor(cv::Mat &stack);
void fillTensor(cv::Mat &src, Tensor &tensor, int start);
cv::Mat getResized(cv::Mat &im);
cv::Mat getStack(cv::Mat *mats);
Tensor getInputImageStack();

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

    std::string path = "/home/nvidia/test/images";
    std::vector<std::string> paths;
    for (const auto & entry : fs::directory_iterator(path)) {
        paths.push_back(entry.path().string());
    }
    int num_images = 0;
    cv::Mat imgs[3];

    auto inputImageStack = getInputImageStack();



    for (const auto & path: paths) {
        cv::Mat test_img = cv::imread(path);
        if (num_images < 3) {
            imgs[num_images] = getResized(test_img);
            num_images++;
        } else {
            imgs[2] = getResized(test_img);
        }
        if (num_images == 3) {
            cv::Mat stack = getStack(imgs);
            Tensor imgTensor = getTensor(stack);
            tensor_dict feedDict = {
                    {"truediv_1:0", imgTensor}
            };
            std::cout << "feed me\n";
            std::vector<tensorflow::Tensor> outputTensors;
            status = session->Run(feedDict, {"egomotion_prediction/pose_exp_net/pose/concat:0"}, {}, &outputTensors);
            std::cout << "feeded\n";
            imgs[0] = imgs[1];
            imgs[1] = imgs[2]; // shift over
            std::cout << outputTensors.size() << std::endl;
            std::cout << outputTensors[0].DebugString() << std::endl;
        }
    }



    return 0;
}

Tensor getInputImageStack() {
    float IMAGENET_MEAN[] = {0.485, 0.456, 0.406};
    float IMAGENET_SD[] = {0.229, 0.224, 0.225};
    auto inputImageStack = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
            {1, 128, 416, 9}));

    auto im_mean = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape{9});
    auto im_std = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape{9});

    auto im_mean_mapped = im_mean.tensor<float, 1>();
    auto im_std_mapped = im_std.tensor<float, 1>();

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            int ind = i * 3 + j;
            im_mean_mapped(ind) = IMAGENET_MEAN[j];
            im_std_mapped(ind) = IMAGENET_SD[j];
        }
    }

    auto im_stack_mapped = inputImageStack.tensor<float, 4>();

    for (int row = 0; row < 128; row++) {
        for (int col = 0; col < 416; col++) {
            for (int d = 0; d < 9; d++) {
                im_stack_mapped(0, row, col, d) = (im_stack_mapped(0, row, col, d) - im_mean_mapped(d)) / im_std_mapped(d);
            }
        }
    }

    return inputImageStack;
}

Tensor getTensor(cv::Mat &stack) {

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                tensorflow::TensorShape({1, stack.rows, stack.cols, 9}));

    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    for (int row = 0; row < stack.rows; row++) {
        for (int col = 0; col < stack.cols; col++) {
            cv::Vec3b pixel = stack.at<cv::Vec3b>(row, col);
            for (int i = 0; i < 9; i++) {
                input_tensor_mapped(0, row, col, i) = pixel[i];
            }
        }
    }

    return input_tensor;
}


cv::Mat getResized(cv::Mat &im) {
    // TODO: convert from 0 to 1 float
    cv::Size size(416,128);//the dst image size,e.g.100x100
    cv::Mat dst;//dst image
    cv::resize(im,dst,size);//resize image
    cv::Mat scaled;
    dst.convertTo(scaled, CV_32F, 1.0 / 255, 0);
    return scaled;
}

cv::Mat getStack(cv::Mat *mats) {
    cv::Mat stack(416, 128, 9);
    for (int img = 0; img < 3; img++) {
        cv::Mat cur = mats[img];
        for (int row = 0; row < 128; row++) {
            for (int col = 0; col < 416; col++) {
                for (int d = 0; d < 3; d++) {
                    stack.at<float>(row, col, d + img * 3) = cur.at<float>(row, col, d);
                }
            }
        }
    }
}