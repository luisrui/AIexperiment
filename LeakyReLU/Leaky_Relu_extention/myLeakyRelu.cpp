#include <torch/extension.h>

#include <iostream>
#include <vector>

std::vector<torch::Tensor> myReLU_forward(
    torch::Tensor input) //,torch::Tensor weights
{
    auto output = torch::max(input, torch::zeros_like(input));

    return {output};
}

std::vector<torch::Tensor> myReLU_backward(
    torch::Tensor grad_output,
    torch::Tensor input //,torch::Tensor weights
)
{
    auto grad_input = torch::mul(grad_output, input.gt(0));
    // auto grad_weights = torch::mm(grad_output.transpose(0, 1), input);

    return {grad_input};
    //, grad_weights;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &myReLU_forward, "myLeakyReLu forward");
    m.def("backward", &myReLU_backward, "myLeakyReLu backward");
}
