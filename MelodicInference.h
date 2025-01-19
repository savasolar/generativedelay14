#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>

#include <torch/script.h>
#include <unordered_map>

//#include <RTNeural/RTNeural.h>
//#include <memory>

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
    std::unordered_map<std::string, int64_t> stoi;
    std::unordered_map<int64_t, std::string> itos;

    bool loadTokenMappings(const std::string& path);
    torch::Tensor preprocess(const std::vector<std::string>& tokens);
    std::vector<std::string> postprocess(const torch::Tensor& logits,
        float temperature,
        int topK);


    bool simple_test();


};