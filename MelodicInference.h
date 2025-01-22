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

    std::unordered_map<std::string, int> stoi;
    std::unordered_map<int, std::string> itos;

    // add model instance pointer
    void* model_context;

    bool loadTokenMappings();
    std::vector<float> softmax(const std::vector<float>& logits, float temperature);
    int sampleFromDistribution(const std::vector<float>& probs);


    bool simple_test();

};