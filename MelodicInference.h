#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>
//#include <unordered_map>
//#include <Eigen/Dense>
#include <RTNeural/RTNeural.h>
#include <memory>

class MelodicInference {
public:
    MelodicInference();
    ~MelodicInference();

    bool loadModel();
    std::vector<std::string> generate(const std::vector<std::string>& prompt,
        float temperature = 0.8f,
        int topK = 200);

private:

    std::unique_ptr<RTNeural::Model<float>> model;
    std::unordered_map<std::string, int> stoi;
    std::unordered_map<int, std::string> itos;


    std::vector<float> preprocess(const std::vector<std::string>& tokens);
    std::vector<std::string> postprocess(const std::vector<float>& logits,
        float temperature,
        int topK);
    bool loadTokenMappings(const nlohmann::json& modelJson);
    std::vector<float> applyTemperatureAndTopK(const std::vector<float>& logits,
        float temperature,
        int topK);



    bool simple_test();


};