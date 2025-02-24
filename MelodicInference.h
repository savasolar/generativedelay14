#pragma once
#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <map>
#include <random>

class MelodicInference {
public:
	MelodicInference();
	~MelodicInference();

	// loads the onnx model, returns true on success
	bool loadModel();

	// generates a 32-symbol melody from a 32-symbol prompt.
	std::vector<std::string> generate(const std::vector<std::string>& prompt,
		float temperature = 0.8f,
		int topK_count = 200);

private:
	std::unique_ptr<Ort::Env> env;
	std::unique_ptr<Ort::Session> session;
	std::map<std::string, int64_t> stoi; // string-to-index mappings
	std::map<int64_t, std::string> itos; // index-to-string mappings
	std::mt19937 rng; // random number generator

	bool loadTokenMappings();
	std::vector<int64_t> topK(const std::vector<float>& logits, int k);
	std::vector<float> softmax(const std::vector<float>& logits);
	std::string symbolsToString(const std::vector<std::string>& symbols);
	std::vector<int64_t> tokenize(const std::string& text);
	std::string detokenize(const std::vector<int64_t>& tokens);
	std::vector<std::string> stringToSymbols(const std::string& text);
	int64_t sampleFromLogits(const std::vector<float>& logits, float temperature, int top_k);

};