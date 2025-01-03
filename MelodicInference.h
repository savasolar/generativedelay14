#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

class MelodicInference {
public:
	MelodicInference();
	~MelodicInference();

	bool loadModel();
	std::vector<std::string> generate(const std::vector<std::string>& prompt, float temperature = 0.8f, int topK = 200);

private:
	std::vector<float> embedding_forward(const std::vector<int>& input_tokens);

	bool test_embedding_simple();

	struct ModelWeights {
		std::vector<float> token_embedding;
		std::vector<float> position_embedding;
		std::vector<float> attention_qkv;
		std::vector<float> attention_bias;
		std::vector<float> lstm_ih;
		std::vector<float> lstm_hh;
		std::vector<float> lstm_bias;
		std::vector<float> output;
		std::vector<float> output_bias;
	};

	struct ModelConfig {
		int vocab_size;
		int embedding_dim;
		int hidden_size;
		int num_heads;
		bool bidirectional;
	};

	ModelWeights weights;
	ModelConfig config;
	std::unordered_map<std::string, int> tokenToIdx;
	std::unordered_map<int, std::string> idxToToken;

	bool loadNPZ(const char* path);
	bool validateWeights();

	bool loadFromBinaryData();

	std::vector<float> add_position_embeddings(std::vector<float>& token_embeddings, size_t seq_len);
	std::vector<size_t> generate_position_indices(size_t seq_len);

	bool test_position_embeddings();
};