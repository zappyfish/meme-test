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
void fillTensor(cv::Mat &src, Tensor &tensor, int startCol);

int main() {
    // set up your input paths
    const string pathToGraph = "model-199160.meta";
    const string checkpointPath = "model-199160";


    auto session = NewSession(SessionOptions());
    if (session == nullptr) {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

// Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok()) {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

// Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if (!status.ok()) {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

// Read weights from the saved checkpoint
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;

    status = session->Run(
            {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);
    if (!status.ok()) {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

// and run the inference to your liking

    std::vector<tensorflow::Tensor> outputTensors;

    std::string path = "/Users/liamkelly/vadl/nvidia/scripts/post-processing/pre-processed-data/feb28_5/images";
    std::vector<std::string> paths;
    for (const auto & entry : fs::directory_iterator(path)) {
        paths.push_back(entry.path().string());
    }
    int num_images = 0;
    cv::Mat imgs[3];

    for (const auto & path: paths) {
        cv::Mat test_img = cv::imread(path);
        imgs[num_images] = test_img;
        if (num_images < 3) {
            num_images++;
        }
        if (num_images == 3) {
            Tensor imgTensor = getTensor(imgs[0], imgs[1], imgs[2]);
            tensor_dict feedDict = {
                    {"input", imgTensor}
            };
            status = session->Run(feedDict, {}, {}, &outputTensors);
            imgs[0] = imgs[1];
            imgs[1] = imgs[2]; // shift over
        }
    }



    return 0;
}

Tensor getTensor(cv::Mat &inp1, cv::Mat &inp2, cv::Mat &inp3) {

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT,
                                tensorflow::TensorShape({1, inp1.rows, inp1.cols * 3, inp1.depth()}));

    fillTensor(inp1, input_tensor, 0);
    fillTensor(inp2, input_tensor, inp1.cols);
    fillTensor(inp3, input_tensor, 2 * inp1.cols);

    return input_tensor;
}

void fillTensor(cv::Mat &src, Tensor &tensor, int startCol) {
    auto input_tensor_mapped = tensor.tensor<float, 4>();
    for (int row = 0; row < src.rows; row++) {
        for (int col = 0; col < src.cols; col++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(row, col);

            input_tensor_mapped(0, row, col + startCol, 0) = pixel[2];
            input_tensor_mapped(0, row, col + startCol, 1) = pixel[1];
            input_tensor_mapped(0, row, col + startCol, 2) = pixel[0];

        }
    }
}
