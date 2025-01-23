#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <map>
#include <random>
#include <Eigen/Dense>
#include <fstream>
#include "nlohmann/json.hpp"

class MelodicInference {
public:
    MelodicInference();
    ~MelodicInference();

    bool loadModel();
    std::vector<std::string> generate(const std::vector<std::string>& prompt,
        float temperature = 0.8f,
        int topK_count = 200);

private:

    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;

    std::map<std::string, int64_t> stoi;
    std::map<int64_t, std::string> itos;
    std::mt19937 rng;

    bool loadTokenMappings();
    std::vector<int64_t> topK(const std::vector<float>& logits, int k);
    float softmax(float val, const std::vector<float>& vals);

};