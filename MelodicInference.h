#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>

#include <torch/script.h>


class MelodicInference {
public:
    MelodicInference();
    ~MelodicInference();

    bool loadModel();
    std::vector<std::string> generate(const std::vector<std::string>& prompt,
        float temperature = 0.8f,
        int topK = 200);

private:

    torch::jit::script::Module model;
    torch::jit::script::Module tokenMappings;

    bool simple_test();

};