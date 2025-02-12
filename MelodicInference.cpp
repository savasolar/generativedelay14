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
//    // convert prompt to string and process character by character
//    std::string promptStr;
//    for (const auto& token : prompt) {
//        promptStr += token + " ";
//    }
//
//    std::vector<int64_t> generatedTokens;
//    for (char c : promptStr) {
//        std::string charStr(1, c);
//        if (stoi.find(charStr) == stoi.end()) {
//            DBG("Character not found in mapping: " + String(charStr));
//            continue; // skip invalid characters instead of returning
//        }
//        generatedTokens.push_back(stoi[charStr]);
//    }
//
//    // generate tokens autoregressively
//    for (int i = 0; i < 128; i++) {
//        // Take last 32 tokens or pad if less
//        std::vector<int64_t> inputIds;
//        if (generatedTokens.size() > 32) {
//            inputIds = std::vector<int64_t>(
//                generatedTokens.end() - 32,
//                generatedTokens.end()
//            );
//        }
//        else {
//            inputIds = generatedTokens;
//            inputIds.resize(32, 0); // pad to 32
//        }
//
//        std::vector<int64_t> inputShape = { 1, 32 };
//
//        // Run inference
//        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
//            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
//        Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
//            memoryInfo, inputIds.data(), inputIds.size(), inputShape.data(), inputShape.size());
//
//        const char* inputNames[] = { "input" };
//        const char* outputNames[] = { "output" };
//        auto outputTensors = session->Run(
//            Ort::RunOptions{ nullptr },
//            inputNames, &inputTensor, 1,
//            outputNames, 1);
//
//        // Get logits for last position only
//        float* logitsData = outputTensors[0].GetTensorMutableData<float>();
//        const size_t vocabSize = stoi.size();
//        std::vector<float> lastLogits(
//            logitsData + (31 * vocabSize),
//            logitsData + (32 * vocabSize)
//        );
//
//        // Apply temperature and sample
//        for (auto& logit : lastLogits) {
//            logit /= temperature;
//        }
//
//        auto topkIndices = topK(lastLogits, topK_count);
//        std::vector<float> probs;
//        probs.reserve(topK_count);
//        for (auto idx : topkIndices) {
//            probs.push_back(softmax(lastLogits[idx], lastLogits));
//        }
//
//        std::discrete_distribution<> dist(probs.begin(), probs.end());
//        int64_t nextToken = topkIndices[dist(rng)];
//
//        // Add new token to sequence
//        generatedTokens.push_back(nextToken);
//    }
//
//    // convert generated tokens to text and process into output format
//    std::string generatedChars;
//    for (size_t i = promptStr.length(); i < generatedTokens.size(); i++) {
//        generatedChars += itos[generatedTokens[i]];
//    }
//
//    // split into tokens and take first 32
//    std::vector<std::string> output;
//    std::istringstream stream(generatedChars);
//    std::string token;
//    while (stream >> token && output.size() < 32) {
//        output.push_back(token);
//    }
//
//    // pad with "_" if needed
//    while (output.size() < 32) {
//        output.push_back("_");
//    }
//
//    DBG("Generate complete");
//    return output;
//
//
//}

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

    // Join prompt vector into space-separated string
    std::string promptStr;
    for (const auto& token : prompt) {
        if (!promptStr.empty()) promptStr += " ";
        promptStr += token;
    }
    DBG("Prompt string: " + String(promptStr));

    // Convert string to tokens character by character exactly like Python:
    // tokens = [token_mappings['stoi'][c] for c in prompt]
    std::vector<int64_t> tokens;
    for (char c : promptStr) {
        std::string charStr(1, c);
        if (stoi.find(charStr) != stoi.end()) {
            tokens.push_back(stoi[charStr]);
        }
    }

    // Create 2D array like numpy does
    std::vector<std::vector<int64_t>> generated_tokens = { tokens };

    // Generation loop
    for (int i = 0; i < 128; i++) {
        // Get last 32 tokens or whole sequence if shorter
        std::vector<int64_t> input_tokens;
        if (generated_tokens[0].size() > 32) {
            input_tokens = std::vector<int64_t>(
                generated_tokens[0].end() - 32,
                generated_tokens[0].end()
            );
        }
        else {
            input_tokens = generated_tokens[0];
        }

        // Create input tensor
        std::vector<int64_t> inputShape = { 1, static_cast<int64_t>(input_tokens.size()) };
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, input_tokens.data(), input_tokens.size(), inputShape.data(), inputShape.size());

        // Run inference
        const char* inputNames[] = { "input" };
        const char* outputNames[] = { "output" };
        auto outputTensors = session->Run(
            Ort::RunOptions{ nullptr },
            inputNames, &inputTensor, 1,
            outputNames, 1
        );

        // Get last logits
        float* logitsData = outputTensors[0].GetTensorMutableData<float>();
        const size_t vocabSize = stoi.size();
        std::vector<float> lastLogits(
            logitsData + ((input_tokens.size() - 1) * vocabSize),
            logitsData + (input_tokens.size() * vocabSize)
        );

        // Apply temperature
        for (auto& logit : lastLogits) {
            logit /= temperature;
        }

        // Get top K and apply softmax
        auto topkIndices = topK(lastLogits, topK_count);
        float maxLogit = *std::max_element(lastLogits.begin(), lastLogits.end());

        std::vector<float> probs(topkIndices.size());
        float sum = 0.0f;
        for (size_t j = 0; j < topkIndices.size(); j++) {
            probs[j] = std::exp(lastLogits[topkIndices[j]] - maxLogit);
            sum += probs[j];
        }
        for (auto& p : probs) {
            p /= sum;
        }

        // Sample next token
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        int64_t next_token = topkIndices[dist(rng)];
        generated_tokens[0].push_back(next_token);
    }

    // Convert generated tokens back to characters and join them
    std::string generated_text;
    for (size_t i = tokens.size(); i < generated_tokens[0].size(); i++) {
        if (itos.find(generated_tokens[0][i]) != itos.end()) {
            generated_text += itos[generated_tokens[0][i]];  // Just concatenate characters
        }
    }

    // Split on spaces to get tokens
    std::vector<std::string> output;
    std::stringstream outSS(generated_text);
    std::string token;
    while (outSS >> token && output.size() < 32) {
        output.push_back(token);
    }

    // Pad to 32 tokens if needed
    while (output.size() < 32) {
        output.push_back("_");
    }

    DBG("Generate complete");
    return output;
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
