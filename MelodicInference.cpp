#include "MelodicInference.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <fstream>
#include <cmath>

MelodicInference::MelodicInference() : rng(std::random_device{}()) {
	loadTokenMappings();
}

MelodicInference::~MelodicInference() {}

bool MelodicInference::loadModel() {
	try {
		env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MelodicInference");
		Ort::SessionOptions session_options;
		std::string model_path = "C:/Users/savas/Desktop/2025-02-09-melodic-nanogpt-2-onnx/melodygpt_v02_quantized.ort";
		std::wstring wmodel_path(model_path.begin(), model_path.end());
		session = std::make_unique<Ort::Session>(*env, wmodel_path.c_str(), session_options);
		DBG("Model loaded successfully");
		return true;
	}
	catch (const Ort::Exception& e) {
		DBG("Failed to load model: " + juce::String(e.what()));
		return false;
	}
}

bool MelodicInference::loadTokenMappings() {
	std::string path = "C:/Users/savas/Desktop/2025-02-09-melodic-nanogpt-2-onnx/token_mappings.json";
	std::ifstream file(path);
	if (!file.is_open()) {
		DBG("Failed to open token mappings");
		return false;
	}
	nlohmann::json j;
	file >> j;
	for (auto& [key, value] : j["stoi"].items()) stoi[key] = value.get<int64_t>();
	for (auto& [key, value] : j["itos"].items()) itos[std::stoll(key)] = value.get<std::string>();
	return true;
}

std::string MelodicInference::symbolsToString(const std::vector<std::string>& symbols) {
	std::ostringstream oss;
	for (size_t i = 0; i < symbols.size(); ++i) {
		if (i > 0) oss << " ";
		oss << symbols[i];
	}
	return oss.str();
}

std::vector<int64_t> MelodicInference::tokenize(const std::string& text) {
	std::vector<int64_t> tokens;
	for (char c : text) tokens.push_back(stoi[std::string(1, c)]);
	return tokens;
}

std::string MelodicInference::detokenize(const std::vector<int64_t>& tokens) {
	std::string text;
	for (int64_t token : tokens) text += itos[token];
	return text;
}

std::vector<std::string> MelodicInference::stringToSymbols(const std::string& text) {
    std::vector<std::string> symbols;
    std::istringstream iss(text);
    std::string symbol;
    while (iss >> symbol) symbols.push_back(symbol);
    return symbols;
}

std::vector<float> MelodicInference::softmax(const std::vector<float>& logits) {
	float max_logit = *std::max_element(logits.begin(), logits.end());
	std::vector<float> exp_logits(logits.size());
	float sum_exp = 0.0f;
	for (size_t i = 0; i < logits.size(); ++i) {
		exp_logits[i] = std::exp(logits[i] - max_logit);
		sum_exp += exp_logits[i];
	}
	for (auto& val : exp_logits) val /= sum_exp;
	return exp_logits;
}

std::vector<int64_t> MelodicInference::topK(const std::vector<float>& logits, int k) {
	std::vector<int64_t> indices(logits.size());
	std::iota(indices.begin(), indices.end(), 0);
	k = std::min(k, static_cast<int>(logits.size()));
	std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
		[&logits](int64_t a, int64_t b) { return logits[a] > logits[b]; });
	return std::vector<int64_t>(indices.begin(), indices.begin() + k);
}

int64_t MelodicInference::sampleFromLogits(const std::vector<float>& logits, float temperature, int top_k) {
	std::vector<float> scaled_logits(logits.size());
	for (size_t i = 0; i < logits.size(); ++i) scaled_logits[i] = logits[i] / temperature;
	auto top_k_indices = topK(scaled_logits, top_k);
	std::vector<float> top_k_logits;
	for (int64_t idx : top_k_indices) top_k_logits.push_back(scaled_logits[idx]);
	auto probs = softmax(top_k_logits);
	std::discrete_distribution<int> dist(probs.begin(), probs.end());
	return top_k_indices[dist(rng)];
}

std::vector<std::string> MelodicInference::generate(const std::vector<std::string>& prompt, float temperature, int top_k) {
	// Ensure the prompt is exactly 32 symbols
	if (prompt.size() != 32) {
		DBG("Error: Prompt must be 32 symbols, got " + juce::String(prompt.size()));
		return std::vector<std::string>(32, "_");
	}

	DBG("Starting generation with prompt: " + symbolsToString(prompt));

	// Convert the prompt to a string and tokenize it using the provided token_mappings.json
	std::string prompt_str = symbolsToString(prompt);
	std::vector<int64_t> tokens = tokenize(prompt_str);

	DBG("Tokenized prompt size: " + juce::String(tokens.size()));

	std::vector<int64_t> generated = tokens;

	// Generate up to 128 additional tokens
	for (int i = 0; i < 128; ++i) {
		// Prepare the context (last 32 tokens, padded with spaces if needed)
		std::vector<int64_t> context(generated.end() - std::min(32ull, generated.size()), generated.end());
		while (context.size() < 32) context.insert(context.begin(), stoi[" "]);

		// Create the input tensor
		Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::vector<int64_t> input_shape = { 1, 32 };
		Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
			memory_info, context.data(), context.size(), input_shape.data(), input_shape.size());

		// Run the model
		const char* input_names[] = { "input" };
		const char* output_names[] = { "output" };
		auto output_tensors = session->Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1, output_names, 1);

		// Extract logits from the model's output
		float* logits_data = output_tensors[0].GetTensorMutableData<float>();
		auto shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

		if (shape.size() == 3 && shape[0] == 1 && shape[1] == 32) {
			size_t vocab_size = shape[2]; // Should be 21 based on token_mappings.json
			// Extract logits for the last position (next token)
			float* last_logits = logits_data + 31 * vocab_size;
			std::vector<float> logits(last_logits, last_logits + vocab_size);

			// Optional: Print logits for debugging
			//std::string logits_str;
			//for (float logit : logits) logits_str += juce::String(logit) + " ";
			//DBG("Logits for step " + juce::String(i) + ": " + logits_str);

			// Sample the next token using the provided temperature and top_k
			int64_t next_token = sampleFromLogits(logits, temperature, top_k);
			DBG("Generated token " + juce::String(i) + ": " + juce::String(next_token));
			generated.push_back(next_token);
		}
		else {
			DBG("Unexpected output shape: [" + juce::String(shape[0]) + ", " +
				(shape.size() > 1 ? juce::String(shape[1]) : "0") + ", " +
				(shape.size() > 2 ? juce::String(shape[2]) : "0") + "]");
			break;
		}
	}

	// Detokenize the generated tokens and convert to symbols
	std::string generated_str = detokenize(std::vector<int64_t>(generated.begin() + tokens.size(), generated.end()));
	auto symbols = stringToSymbols(generated_str);
	symbols.resize(32, "_"); // Ensure exactly 32 symbols
	DBG("Generated melody: " + symbolsToString(symbols));
	return symbols;
}