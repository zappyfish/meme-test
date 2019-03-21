//
// Created by Liam Kelly on 3/21/19.
//

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/platform/env.h"
#include <opencv2/opencv.hpp>

#ifndef CPPNN_EGOMOTION_MODEL_H
#define CPPNN_EGOMOTION_MODEL_H

using namespace tensorflow;

class EgomotionModel {

public:

    EgomotionModel();
    ~EgomotionModel();

private:

    Tensor buildInputImageStack();
    Tensor buildEstEgomotion();
};


#endif //CPPNN_EGOMOTION_MODEL_H
