#include "MelodicInference.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>

MelodicInference::MelodicInference()
    : env_(ORT_LOGGING_LEVEL_WARNING, "MelodicInference"),
    rng_(std::random_device{}())
{
}

MelodicInference::~MelodicInference()
{
    // Session is auto-cleaned by unique_ptr
}

bool MelodicInference::loadModel(const juce::String& modelPath, const juce::String& tokensPath)
{
    try {
        DBG("Loading ONNX model from: " + modelPath);

        juce::File modelFile(modelPath);
        if (!modelFile.existsAsFile()) {
            DBG("Model file not found: " + modelPath);
            return false;
        }

        // Set up session options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Create session
//        session_ = std::make_unique<Ort::Session>(env_, modelFile.getFullPathName().toStdString().c_str(), sessionOptions);

#ifdef _WIN32
// On Windows, use wide character strings
        std::wstring wModelPath = std::wstring(modelFile.getFullPathName().toWideCharPointer());
        session_ = std::make_unique<Ort::Session>(env_, wModelPath.c_str(), sessionOptions);
#else
    // On other platforms, use UTF-8 strings
        session_ = std::make_unique<Ort::Session>(env_, modelFile.getFullPathName().toStdString().c_str(), sessionOptions);
#endif

        // Load token mappings
        loadTokenMappings(tokensPath);

        DBG("Model loaded successfully");
        return true;
    }
    catch (const Ort::Exception& e) {
        DBG("ONNX Runtime error loading model: " + juce::String(e.what()));
        return false;
    }
    catch (const std::exception& e) {
        DBG("Error loading model: " + juce::String(e.what()));
        return false;
    }
}

void MelodicInference::loadTokenMappings(const juce::String& tokensPath)
{
    DBG("Loading token mappings from: " + tokensPath);

    juce::File file(tokensPath);
    if (!file.existsAsFile()) {
        DBG("Token mappings file not found: " + tokensPath);
        return;
    }

    // Parse JSON
    auto jsonStr = file.loadFileAsString();
    juce::var json = juce::JSON::parse(jsonStr);

    if (!json.isObject()) {
        DBG("Failed to parse token mappings JSON");
        return;
    }

    // Clear existing mappings
    stoi_.clear();
    itos_.clear();

    // Load stoi mappings
    if (json.hasProperty("stoi") && json["stoi"].isObject()) {
        auto stoi = json["stoi"].getDynamicObject();
        for (auto& prop : stoi->getProperties()) {
            stoi_[prop.name.toString().toStdString()] = static_cast<int>(prop.value);
        }
    }

    // Load itos mappings
    if (json.hasProperty("itos") && json["itos"].isObject()) {
        auto itos = json["itos"].getDynamicObject();
        for (auto& prop : itos->getProperties()) {
            itos_[std::stoi(prop.name.toString().toStdString())] = prop.value.toString().toStdString();
        }
    }

    DBG("Loaded " + juce::String(stoi_.size()) + " stoi mappings and " +
        juce::String(itos_.size()) + " itos mappings");
}

std::vector<float> MelodicInference::softmax(const std::vector<float>& x)
{
    // Find max value for numerical stability
    float max_val = *std::max_element(x.begin(), x.end());

    // Compute exp(x - max_val) for each element
    std::vector<float> exp_x;
    exp_x.reserve(x.size());
    for (auto val : x) {
        exp_x.push_back(std::exp(val - max_val));
    }

    // Compute sum of exp values
    float sum_exp = std::accumulate(exp_x.begin(), exp_x.end(), 0.0f);

    // Normalize to get softmax
    std::vector<float> softmax_result;
    softmax_result.reserve(x.size());
    for (auto val : exp_x) {
        softmax_result.push_back(val / sum_exp);
    }

    return softmax_result;
}

int MelodicInference::sampleFromLogits(const std::vector<float>& logits, float temperature, int topK)
{
    // Apply temperature
    std::vector<float> scaled_logits;
    scaled_logits.reserve(logits.size());
    for (auto logit : logits) {
        scaled_logits.push_back(logit / temperature);
    }

    // Apply top-k filtering if specified
    if (topK > 0 && topK < static_cast<int>(scaled_logits.size())) {
        // Find values for top K elements
        std::vector<float> sorted_logits = scaled_logits;
        std::partial_sort(sorted_logits.begin(), sorted_logits.begin() + topK, sorted_logits.end(), std::greater<float>());
        float threshold = sorted_logits[topK - 1];

        // Set all values below threshold to negative infinity
        for (auto& logit : scaled_logits) {
            if (logit < threshold) {
                logit = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // Apply softmax
    std::vector<float> probs = softmax(scaled_logits);

    // Sample from distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng_);
}

std::vector<std::string> MelodicInference::generate(const std::vector<std::string>& inputTokens,
    float temperature,
    int topK)
{
    if (!session_) {
        DBG("Model not loaded!");
        return inputTokens;
    }

    try {
        // Convert symbol-level tokens to a string prompt
        std::string promptStr;
        for (const auto& token : inputTokens) {
            promptStr += token + " ";
        }
        if (!promptStr.empty()) {
            promptStr.pop_back(); // Remove trailing space
        }

        DBG("Input prompt: " + juce::String(promptStr));

        // Tokenize the prompt character by character
        std::vector<int64_t> tokens;
        for (char c : promptStr) {
            std::string charStr(1, c);
            auto it = stoi_.find(charStr);
            if (it != stoi_.end()) {
                tokens.push_back(it->second);
            }
            else {
                DBG("Unknown character in input: '" + juce::String(charStr) + "'");
                // Skip unknown characters
            }
        }

        if (tokens.empty()) {
            DBG("No valid tokens in input");
            return inputTokens;
        }

        DBG("Tokenized input with " + juce::String(tokens.size()) + " character tokens");

        // Setup for inference
        std::vector<int64_t> generated_tokens = tokens;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const char* input_names[] = { "input" };
        const char* output_names[] = { "output" };

        // Generate new tokens
        const int max_new_tokens = 128;
        for (int i = 0; i < max_new_tokens; i++) {
            // Use context window of at most 32 tokens
            std::vector<int64_t> input_tokens;
            if (generated_tokens.size() > 32) {
                input_tokens.assign(generated_tokens.end() - 32, generated_tokens.end());
            }
            else {
                input_tokens = generated_tokens;
            }

            // Create input tensor
            std::vector<int64_t> input_shape = { 1, static_cast<int64_t>(input_tokens.size()) };
            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, input_tokens.data(), input_tokens.size(),
                input_shape.data(), input_shape.size());

            // Run inference
            auto output_tensors = session_->Run(
                Ort::RunOptions{ nullptr },
                input_names, &input_tensor, 1,
                output_names, 1);

            // Process output
//            float* output_data = output_tensors[0].GetTensorData<float>();
            const float* output_data = output_tensors[0].GetTensorData<float>();
            auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

            // Extract dimensions from output shape
            int64_t vocab_size = output_shape[2]; // [batch, seq_len, vocab_size]

            // Get logits for the last token (last position in sequence)
            size_t last_pos = input_tokens.size() - 1;
            std::vector<float> last_token_logits(vocab_size);
            for (size_t v = 0; v < vocab_size; v++) {
                last_token_logits[v] = output_data[(last_pos * vocab_size) + v];
            }

            // Sample next token
            int next_token = sampleFromLogits(last_token_logits, temperature, topK);

            // Add to generated tokens
            generated_tokens.push_back(next_token);
        }

        // Convert generated tokens back to text (only the new ones)
        std::string generated_text;
        for (size_t i = tokens.size(); i < generated_tokens.size(); i++) {
            int64_t token = generated_tokens[i];
            auto it = itos_.find(token);
            if (it != itos_.end()) {
                generated_text += it->second;
            }
            else {
                DBG("Unknown token ID in output: " + juce::String(token));
            }
        }

        DBG("Raw generated text: " + juce::String(generated_text));

        // Split the generated text into space-separated tokens
        std::vector<std::string> output_tokens;
        std::istringstream stream(generated_text);
        std::string token;
        while (stream >> token) {
            output_tokens.push_back(token);
        }

        // Take the first 32 tokens, or pad with "_" if needed
        std::vector<std::string> result;
        for (int i = 0; i < 32; i++) {
            if (i < static_cast<int>(output_tokens.size())) {
                result.push_back(output_tokens[i]);
            }
            else {
                result.push_back("_");
            }
        }

        // Debug output
        juce::String result_str;
        for (const auto& token : result) {
//            result_str += token + " ";
            result_str += juce::String(token) + " ";
        }
        DBG("Final generated melody: " + result_str.trimEnd());

        return result;
    }
    catch (const Ort::Exception& e) {
        DBG("ONNX Runtime error during generation: " + juce::String(e.what()));
        return inputTokens;
    }
    catch (const std::exception& e) {
        DBG("Error during generation: " + juce::String(e.what()));
        return inputTokens;
    }
}