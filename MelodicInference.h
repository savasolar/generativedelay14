#pragma once
#include <JuceHeader.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Dense>

class MelodicInference {
public:
    MelodicInference();
    ~MelodicInference();

    bool loadModel();
    std::vector<std::string> generate(const std::vector<std::string>& prompt,
        float temperature = 0.8f,
        int topK = 200);

private:
    struct ModelConfig {
        int vocab_size;
        int embedding_dim;
        int hidden_size;
    };

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


    bool loadConfig(const std::string& filename);
    bool loadTokenEmbeddings(const std::string& filename);
    bool loadPositionEmbeddings(const std::string& filename);

    bool loadAttentionWeights();
    bool loadAttentionBias();


    Eigen::MatrixXf getTokenEmbeddings(const std::vector<int>& input_tokens);
    Eigen::MatrixXf addPositionEmbeddings(const Eigen::MatrixXf& token_embeddings);
    Eigen::MatrixXf computeAttention(const Eigen::MatrixXf& embeddings);
    Eigen::MatrixXf processLSTM(const Eigen::MatrixXf& attention_output);
    Eigen::VectorXf computeLogits(const Eigen::MatrixXf& lstm_output);
    Eigen::VectorXf forward(const std::vector<int>& tokens);





    ModelWeights weights;
    ModelConfig config;
    std::unordered_map<std::string, int> tokenToIdx;
    std::unordered_map<int, std::string> idxToToken;


    bool test_embedding_simple();


    bool test_attention();

};