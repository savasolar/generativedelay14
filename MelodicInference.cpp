#include "MelodicInference.h"

MelodicInference::MelodicInference() : rng(std::random_device{}()) {
    DBG("Starting constructor");
    if (!loadTokenMappings()) {
        DBG("Failed to load tokens in constructor");
    }
    DBG("Constructor complete");
}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
    try {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;

//        std::string model_path = "C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/melodic_model.ort";
        std::string model_path = "C:/Users/savas/Desktop/2025-02-09-melodic-nanogpt-2-onnx/melodygpt_v02_quantized.ort";


        std::wstring wmodel_path(model_path.begin(), model_path.end());
        session = std::make_unique<Ort::Session>(
            *env,
            wmodel_path.c_str(),
            session_options
        );

        DBG("Model loaded successfully!");
        return true;
    }
    catch (const Ort::Exception& e) {
        DBG("Failed to load model: " + String(e.what()));
        return false;
    }
}



bool MelodicInference::loadTokenMappings() {
    try {
//        std::string path = "C:/Users/savas/Documents/JUCE Projects/generativedelay14/token_mappings.json";
        std::string path = "C:/Users/savas/Desktop/2025-02-09-melodic-nanogpt-2-onnx/token_mappings.json";
        DBG("Loading tokens from: " + String(path));
        std::ifstream file(path);
        nlohmann::json j;
        file >> j;

        DBG("Loaded JSON, parsing maps...");
        auto stoiMap = j["stoi"];
        for (auto& [key, value] : stoiMap.items()) {
            stoi[key] = value.get<int64_t>();
        }
        DBG("Loaded " + String(stoi.size()) + " stoi mappings");

        auto itosMap = j["itos"];
        for (auto& [key, value] : itosMap.items()) {
            itos[std::stoll(key)] = value.get<std::string>();
        }
        DBG("Loaded " + String(itos.size()) + " itos mappings");
        return true;
    }
    catch (const std::exception& e) {
        DBG("Failed to load token mappings: " + String(e.what()));
        return false;
    }
}

std::vector<std::string> MelodicInference::generate(
    const std::vector<std::string>& prompt,
    float temperature,
    int topK_count)
{
    DBG("Starting generate");
    if (!session) {
        DBG("No session - call loadModel() first");
        return {};
    }

    //std::vector<std::string> output;
    //output.reserve(32);

    //DBG("Converting prompt tokens: " + String(prompt.size()));
    //std::vector<int64_t> inputIds;
    //for (const auto& token : prompt) {
    //    if (stoi.find(token) == stoi.end()) {
    //        DBG("Token not found in mapping: " + String(token));
    //        return {};
    //    }
    //    inputIds.push_back(stoi[token]);
    //}

    //const int64_t seqLen = 32;
    //std::vector<int64_t> inputShape = { 1, seqLen };
    //DBG("Resizing input to " + String(seqLen));
    //inputIds.resize(seqLen, 0);

    //Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
    //    OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    //Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
    //    memoryInfo, inputIds.data(), inputIds.size(), inputShape.data(), inputShape.size());

    //const char* inputNames[] = { "input" };
    //const char* outputNames[] = { "output" };
    //auto outputTensors = session->Run(
    //    Ort::RunOptions{ nullptr },
    //    inputNames, &inputTensor, 1,
    //    outputNames, 1);

    //float* logitsData = outputTensors[0].GetTensorMutableData<float>();
    //const size_t vocabSize = stoi.size();
    //DBG("Vocab size: " + String(vocabSize));

    //for (int pos = 0; pos < 32; pos++) {
    //    std::vector<float> posLogits(logitsData + pos * vocabSize,
    //        logitsData + (pos + 1) * vocabSize);

    //    DBG("Applying temperature: " + String(temperature));
    //    for (auto& logit : posLogits) {
    //        logit /= temperature;
    //    }

    //    DBG("Getting top " + String(topK_count) + " indices");
    //    auto topkIndices = topK(posLogits, topK_count);

    //    std::vector<float> probs;
    //    probs.reserve(topK_count);
    //    for (auto idx : topkIndices) {
    //        probs.push_back(softmax(posLogits[idx], posLogits));
    //    }

    //    DBG("Sampling from distribution");
    //    std::discrete_distribution<> dist(probs.begin(), probs.end());
    //    int64_t nextToken = topkIndices[dist(rng)];

    //    DBG("Selected token: " + String(nextToken));
    //    output.push_back(itos[nextToken]);
    //}

    //DBG("Generate complete");
    //return output;


    // changing code for ort model v2


}


std::vector<int64_t> MelodicInference::topK(const std::vector<float>& logits, int k) {
    k = std::min(k, static_cast<int>(logits.size()));
    std::vector<std::pair<float, int64_t>> pairs;
    for (size_t i = 0; i < logits.size(); i++) {
        pairs.push_back({ logits[i], i });
    }

    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int64_t> indices;
    indices.reserve(k);
    for (int i = 0; i < k; i++) {
        indices.push_back(pairs[i].second);
    }
    return indices;
}

float MelodicInference::softmax(float val, const std::vector<float>& vals) {
    float maxVal = *std::max_element(vals.begin(), vals.end());
    float exp_val = std::exp(val - maxVal);
    float sum = 0;
    for (auto v : vals) {
        sum += std::exp(v - maxVal);
    }
    return exp_val / sum;
}
