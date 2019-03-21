//
// Created by Liam Kelly on 3/21/19.
//

#include "egomotion_model.h"

EgomotionModel::EgomotionModel() {
    Scope root = Scope::NewRootScope();
    Scope linear = root.NewSubScope("egomotion");
}

Tensor EgomotionModel::buildInputImageStack() {
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

Tensor EgomotionModel::buildEstEgomotion() {
    auto estEgomotion = tensorflow::Tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape(
            {1, 128, 416, 9}));
    return estEgomotion;
}