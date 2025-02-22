#include "MelodicInference.h"

//MelodicInference::MelodicInference() : rng(std::random_device{}()) {
//    DBG("Starting constructor");
//    if (!loadTokenMappings()) {
//        DBG("Failed to load tokens in constructor");
//    }
//    DBG("Constructor complete");
//}

MelodicInference::MelodicInference() : rng(42) { // set seed to 42
    if (!loadTokenMappings()) {
        DBG("Failed to load token mappings");
    }
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




std::vector<std::string> MelodicInference::generate(const std::vector<std::string>& prompt, float temperature, int topK_count) {

}



// Helper: Top-k indices
std::vector<int64_t> MelodicInference::topK(const std::vector<float>& logits, int k) {

}




float MelodicInference::softmax(float val, const std::vector<float>& vals) {

}
