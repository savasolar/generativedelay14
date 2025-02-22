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

//std::vector<std::string> MelodicInference::generate(
//    const std::vector<std::string>& prompt,
//    float temperature,
//    int topK_count)
//{
//    DBG("Starting generate");
//    if (!session) {
//        DBG("No session - call loadModel() first");
//        return {};
//    }
//
//    // Join prompt vector into space-separated string
//    std::string promptStr;
//    for (const auto& token : prompt) {
//        if (!promptStr.empty()) promptStr += " ";
//        promptStr += token;
//    }
//    DBG("Prompt string: " + String(promptStr));
//
//    // Convert string to tokens character by character exactly like Python:
//    // tokens = [token_mappings['stoi'][c] for c in prompt]
//    std::vector<int64_t> tokens;
//    for (char c : promptStr) {
//        std::string charStr(1, c);
//        if (stoi.find(charStr) != stoi.end()) {
//            tokens.push_back(stoi[charStr]);
//        }
//    }
//
//    // Create 2D array like numpy does
//    std::vector<std::vector<int64_t>> generated_tokens = { tokens };
//
//    // Generation loop
//    for (int i = 0; i < 128; i++) {
//        // Get last 32 tokens or whole sequence if shorter
//        std::vector<int64_t> input_tokens;
//        if (generated_tokens[0].size() > 32) {
//            input_tokens = std::vector<int64_t>(
//                generated_tokens[0].end() - 32,
//                generated_tokens[0].end()
//            );
//        }
//        else {
//            input_tokens = generated_tokens[0];
//        }
//
//        // Create input tensor
//        std::vector<int64_t> inputShape = { 1, static_cast<int64_t>(input_tokens.size()) };
//        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
//            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//        Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
//            memoryInfo, input_tokens.data(), input_tokens.size(), inputShape.data(), inputShape.size());
//
//        // Run inference
//        const char* inputNames[] = { "input" };
//        const char* outputNames[] = { "output" };
//        auto outputTensors = session->Run(
//            Ort::RunOptions{ nullptr },
//            inputNames, &inputTensor, 1,
//            outputNames, 1
//        );
//
//        // Get last logits
//        float* logitsData = outputTensors[0].GetTensorMutableData<float>();
//        const size_t vocabSize = stoi.size();
//        std::vector<float> lastLogits(
//            logitsData + ((input_tokens.size() - 1) * vocabSize),
//            logitsData + (input_tokens.size() * vocabSize)
//        );
//
//        // Apply temperature
//        for (auto& logit : lastLogits) {
//            logit /= temperature;
//        }
//
//        // Get top K and apply softmax
//        auto topkIndices = topK(lastLogits, topK_count);
//        float maxLogit = *std::max_element(lastLogits.begin(), lastLogits.end());
//
//        std::vector<float> probs(topkIndices.size());
//        float sum = 0.0f;
//        for (size_t j = 0; j < topkIndices.size(); j++) {
//            probs[j] = std::exp(lastLogits[topkIndices[j]] - maxLogit);
//            sum += probs[j];
//        }
//        for (auto& p : probs) {
//            p /= sum;
//        }
//
//        // Sample next token
//        std::discrete_distribution<> dist(probs.begin(), probs.end());
//        int64_t next_token = topkIndices[dist(rng)];
//        generated_tokens[0].push_back(next_token);
//    }
//
//    // Convert generated tokens back to characters and join them
//    std::string generated_text;
//    for (size_t i = tokens.size(); i < generated_tokens[0].size(); i++) {
//        if (itos.find(generated_tokens[0][i]) != itos.end()) {
//            generated_text += itos[generated_tokens[0][i]];  // Just concatenate characters
//        }
//    }
//
//    // Split on spaces to get tokens
//    std::vector<std::string> output;
//    std::stringstream outSS(generated_text);
//    std::string token;
//    while (outSS >> token && output.size() < 32) {
//        output.push_back(token);
//    }
//
//    // Pad to 32 tokens if needed
//    while (output.size() < 32) {
//        output.push_back("_");
//    }
//
//    DBG("Generate complete");
//    return output;
//}
//
//
//std::vector<int64_t> MelodicInference::topK(const std::vector<float>& logits, int k) {
//    k = std::min(k, static_cast<int>(logits.size()));
//    std::vector<std::pair<float, int64_t>> pairs;
//    for (size_t i = 0; i < logits.size(); i++) {
//        pairs.push_back({ logits[i], i });
//    }
//
//    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
//        [](const auto& a, const auto& b) { return a.first > b.first; });
//
//    std::vector<int64_t> indices;
//    indices.reserve(k);
//    for (int i = 0; i < k; i++) {
//        indices.push_back(pairs[i].second);
//    }
//    return indices;
//}




std::vector<std::string> MelodicInference::generate(const std::vector<std::string>& prompt, float temperature, int topK_count) {
    if (!session) {
        DBG("No session - call loadModel() first");
        return {};
    }

    // Join prompt into a single string, exactly as VST intends
    std::string promptStr;
    for (size_t i = 0; i < prompt.size(); ++i) {
        promptStr += prompt[i];
        if (i < prompt.size() - 1) promptStr += " ";
    }
    DBG("Prompt: " + String(promptStr)); // e.g., "60 - _ _"

    // Tokenize character-by-character like Python
    std::vector<int64_t> tokens;
    for (char c : promptStr) {
        std::string charStr(1, c);
        if (stoi.find(charStr) != stoi.end()) {
            tokens.push_back(stoi[charStr]);
        }
        else {
            DBG("Unmapped char: " + String(charStr));
        }
    }
    DBG("Token count: " + String(tokens.size()));

    std::vector<int64_t> generated_tokens = tokens;
    for (int i = 0; i < 128; i++) { // Generate 128 tokens like Python
        std::vector<int64_t> input_tokens;
        if (generated_tokens.size() > 32) {
            input_tokens.assign(generated_tokens.end() - 32, generated_tokens.end());
        }
        else {
            input_tokens = generated_tokens;
        }
        std::vector<int64_t> inputShape = { 1, static_cast<int64_t>(input_tokens.size()) };
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, input_tokens.data(), input_tokens.size(), inputShape.data(), inputShape.size());

        const char* inputNames[] = { "input" };
        const char* outputNames[] = { "output" };
        auto outputTensors = session->Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1);

        float* logitsData = outputTensors[0].GetTensorMutableData<float>();
        size_t vocabSize = stoi.size();
        std::vector<float> lastLogits(logitsData + (input_tokens.size() - 1) * vocabSize, logitsData + input_tokens.size() * vocabSize);

        // Temperature scaling
        for (auto& logit : lastLogits) {
            logit /= temperature;
        }
        // Top-k filtering
        auto topkIndices = topK(lastLogits, topK_count);
        std::vector<float> filtered_logits(lastLogits.size(), -std::numeric_limits<float>::infinity());
        for (auto idx : topkIndices) {
            filtered_logits[idx] = lastLogits[idx];
        }
        // Softmax
        float maxLogit = *std::max_element(filtered_logits.begin(), filtered_logits.end());
        std::vector<float> probs(filtered_logits.size());
        float sum = 0.0f;
        for (size_t j = 0; j < probs.size(); ++j) {
            probs[j] = std::exp(filtered_logits[j] - maxLogit);
            sum += probs[j];
        }
        for (auto& p : probs) {
            p /= sum;
        }
        // Sample with seeded RNG
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int64_t next_token = dist(rng);
        generated_tokens.push_back(next_token);
    }

    std::string generated_text;
    for (size_t i = tokens.size(); i < generated_tokens.size(); ++i) { // Skip prompt
        if (itos.find(generated_tokens[i]) != itos.end()) {
            generated_text += itos[generated_tokens[i]];
        }
    }
    DBG("Generated text: " + String(generated_text));

    std::vector<std::string> output;
    std::istringstream iss(generated_text);
    std::string token;
    while (iss >> token && output.size() < 32) {
        output.push_back(token);
    }
    while (output.size() < 32) {
        output.push_back("_");
    }
    DBG("Output size: " + String(output.size()));
    return output;
}



// Helper: Top-k indices
std::vector<int64_t> MelodicInference::topK(const std::vector<float>& logits, int k) {
    std::vector<std::pair<float, int64_t>> pairs;
    for (size_t i = 0; i < logits.size(); ++i) {
        pairs.emplace_back(logits[i], i);
    }
    std::partial_sort(pairs.begin(), pairs.begin() + std::min(k, static_cast<int>(logits.size())), pairs.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<int64_t> indices;
    for (int i = 0; i < std::min(k, static_cast<int>(logits.size())); ++i) {
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
