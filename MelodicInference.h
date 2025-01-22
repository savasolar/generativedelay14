#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>
//#include <onnxruntime_cxx_api.h>
//#include <nlohmann/json.hpp>
#include "model.h.h"


class MelodicInference {
public:
    MelodicInference();
    ~MelodicInference();

    bool loadModel();
    std::vector<std::string> generate(const std::vector<std::string>& prompt,
        float temperature = 0.8f,
        int topK = 200);

private:


    bool simple_test();

};