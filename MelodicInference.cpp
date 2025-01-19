#include "MelodicInference.h"
#include <fstream>
#include <random>


MelodicInference::MelodicInference() {}
MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
    DBG("Loading model...");
    juce::String modelPath = "C:/Users/savas/Documents/JUCE Projects/generativedelay14/Model/model_rtneural.json";
    DBG("Model path: " + modelPath);

    std::ifstream jsonStream(modelPath.toStdString());
    if (!jsonStream) {
        DBG("Failed to open model file");
        return false;
    }

    try {
        // Read entire file content once
        nlohmann::json modelJson;
        jsonStream >> modelJson;
        DBG("Loaded JSON");


        DBG("JSON content preview:");
        DBG(modelJson.dump().substr(0, 500)); // First 500 chars


        if (!loadTokenMappings(modelJson)) {
            DBG("Failed to load token mappings");
            return false;
        }
        DBG("Loaded token mappings");

        // Create new stream for RTNeural
        std::ifstream modelStream(modelPath.toStdString());
        model = RTNeural::json_parser::parseJson<float>(modelStream);
        if (!model) {
            DBG("Failed to parse model");
            return false;
        }
        DBG("Parsed model");

        model->reset();
        DBG("Model loaded successfully");
        return true;
    }
    catch (const std::exception& e) {
        DBG("Exception during model load: " + String(e.what()));
        return false;
    }
}




bool MelodicInference::loadTokenMappings(const nlohmann::json& modelJson) {
    try {
        DBG("Checking for token_mapping...");
        if (!modelJson.contains("token_mapping")) {
            DBG("No token_mapping found in JSON");
            return false;
        }

        auto mappings = modelJson["token_mapping"];
        DBG("Got mappings object");

        DBG("Loading stoi mappings...");
        if (!mappings.contains("stoi")) {
            DBG("No stoi found in mappings");
            return false;
        }
        for (const auto& [token, idx] : mappings["stoi"].items()) {
            DBG("Loading token: " + juce::String(token) + " -> " + juce::String(idx.get<int>()));
            stoi[token] = idx.get<int>();
        }

        DBG("Loading itos mappings...");
        if (!mappings.contains("itos")) {
            DBG("No itos found in mappings");
            return false;
        }
        for (const auto& [idx_str, token] : mappings["itos"].items()) {
            DBG("Loading index: " + juce::String(idx_str) + " -> " + juce::String(token.get<std::string>()));
            itos[std::stoi(idx_str)] = token.get<std::string>();
        }

        return true;
    }
    catch (const std::exception& e) {
        DBG("Exception in loadTokenMappings: " + juce::String(e.what()));
        return false;
    }
}


std::vector<float> MelodicInference::preprocess(const std::vector<std::string>& tokens) {
    std::vector<float> input;
    input.reserve(tokens.size());
    for (const auto& token : tokens)
        input.push_back(static_cast<float>(stoi[token]));
    return input;
}

std::vector<float> MelodicInference::applyTemperatureAndTopK(
    const std::vector<float>& logits,
    float temperature,
    int topK)
{
    std::vector<float> probs = logits;

    // Apply temperature
    for (auto& x : probs)
        x /= temperature;

    // Apply top-k
    if (topK > 0 && topK < (int)probs.size()) {
        std::vector<std::pair<float, size_t>> pairs;
        for (size_t i = 0; i < probs.size(); ++i)
            pairs.emplace_back(probs[i], i);

        std::partial_sort(pairs.begin(),
            pairs.begin() + topK,
            pairs.end(),
            std::greater<>());

        std::fill(probs.begin(), probs.end(), 0.0f);
        for (int i = 0; i < topK; ++i)
            probs[pairs[i].second] = pairs[i].first;
    }

    // Softmax
    float max_val = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (auto& x : probs) {
        x = std::exp(x - max_val);
        sum += x;
    }
    for (auto& x : probs)
        x /= sum;

    return probs;
}

std::vector<std::string> MelodicInference::postprocess(
    const std::vector<float>& probs,
    float temperature,
    int topK)
{
    std::vector<std::string> output;
    std::random_device rd;
    std::mt19937 gen(rd());

    // Sample from probabilities
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(gen);
    output.push_back(itos[idx]);

    return output;
}



//std::vector<std::string> MelodicInference::generate(
//    const std::vector<std::string>& prompt,
//    float temperature,
//    int topK)
//{
//    if (!model)
//        return {};
//
//    model->reset();
//    auto input = preprocess(prompt);
//    float output = model->forward(input.data());
//
//    std::vector<float> logits{ output };
//    auto processed = applyTemperatureAndTopK(logits, temperature, topK);
//    return postprocess(processed, temperature, topK);
//}


std::vector<std::string> MelodicInference::generate(
    const std::vector<std::string>& prompt,
    float temperature,
    int topK)
{
    if (!model) {
        DBG("Model not loaded!");
        return {};
    }

    DBG("Generating from prompt:");
    for (const auto& token : prompt)
        DBG(token);

    model->reset();
    auto input = preprocess(prompt);

    DBG("Preprocessed input:");
    for (const auto& val : input)
        DBG(String(val));

    float output = model->forward(input.data());
    DBG("Raw model output: " + String(output));

    std::vector<float> logits{ output };
    auto processed = applyTemperatureAndTopK(logits, temperature, topK);

    DBG("After temperature/topK:");
    for (const auto& prob : processed)
        DBG(String(prob));

    auto result = postprocess(processed, temperature, topK);

    DBG("Generated tokens:");
    for (const auto& token : result)
        DBG(token);

    return result;
}





bool MelodicInference::simple_test() {

    // implement generation test

    return {};
}

