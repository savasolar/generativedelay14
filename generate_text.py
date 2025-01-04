# terminal:
# pip install torch numpy transformers datasets tiktoken requests wandb flask

from flask import Flask, request, jsonify
import json
import torch
import torch.nn.functional as F
from model import MelodicStudent
import tiktoken
import pickle
import numpy as np

app = Flask(__name__)

checkpoint = torch.load('checkpoint_e8_b168000.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
model = MelodicStudent(
    vocab_size=checkpoint['vocab_size'],
    embedding_dim=checkpoint['embedding_dim'],
    hidden_size=checkpoint['hidden_size']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Update tokenizer loading:
stoi = checkpoint['token2idx']
itos = checkpoint['idx2token']

# after model loading:
# print("\nPosition embedding layout:")
# pos_emb = model.position_embedding.weight.detach().cpu().numpy()
# print("Shape:", pos_emb.shape)
# print("First position (first 3 values):", pos_emb[0, :3])
# print("Raw weights (first 3):", model.position_embedding.weight.detach().cpu().numpy()[:3])

test_tokens = torch.tensor([[60, 45]]) 
token_emb = model.embedding(test_tokens)
pos_emb = model.position_embedding(torch.arange(2))

print("\nComplete verification:")
print("1. Token embedding at pos 0:", token_emb[0, 0, 0].item())
print("2. Position embedding at pos 0:", pos_emb[0, 0].item())
print("3. Sum:", token_emb[0, 0, 0].item() + pos_emb[0, 0].item())

# print("Token embedding first 5:", token_emb[0, 0, :5].tolist())
# print("Position 0 embedding first 5:", pos_emb[0, :5].tolist())  
# print("Combined first 5:", (token_emb + pos_emb.unsqueeze(0))[0, 0, :5].tolist())

# print("Token embedding first value:", token_emb[0, 0, 0].item())
# print("Position embedding first value:", pos_emb[0, 0].item())

# working function for exporting binary weights
def export_binary_weights():
    # Debug prints
    # Debug position embeddings specifically
    pos_emb = model.position_embedding.weight.detach().cpu().numpy()
    print("\nPosition Embedding Debug:")
    print("Shape:", pos_emb.shape)
    print("First position (raw numpy):", pos_emb[0, :3])
    print("Memory layout:", pos_emb.strides)
    print("First 6 values as written to file:", pos_emb.flatten()[:6])
    
    weights = {
        'token_embedding': model.embedding.weight,
        'position_embedding': model.position_embedding.weight,
        'attention_qkv': model.attention.in_proj_weight,
        'attention_bias': model.attention.in_proj_bias,
        'lstm_ih': model.lstm.weight_ih_l0,
        'lstm_hh': model.lstm.weight_hh_l0,
        'lstm_bias': model.lstm.bias_ih_l0,
        'output': model.output.weight,
        'output_bias': model.output.bias
    }

    with open('model_weights.bin', 'wb') as f:
        # Existing config writing
        for value in [checkpoint['vocab_size'], checkpoint['embedding_dim'], checkpoint['hidden_size']]:
            f.write(value.to_bytes(4, byteorder='little'))
        
        # Existing token mappings
        for token, idx in stoi.items():
            token_bytes = token.encode('utf-8')
            f.write(len(token_bytes).to_bytes(4, byteorder='little'))
            f.write(token_bytes)
            f.write(idx.to_bytes(4, byteorder='little'))

        # Write weights with debug for first tensor
        for i, (name, tensor) in enumerate(weights.items()):
            data = tensor.detach().cpu().numpy()
            # if i == 0:  # token_embedding
            #     print("Matrix shape before transpose:", data.shape)
            #     data = data.T
            #     print("Matrix shape after transpose:", data.shape)
            data.tofile(f)

print("Token 60 embedding export order check:")
emb = model.embedding.weight[60].detach().cpu().numpy()
print("First 5 values in native order:", emb[:5])

# Call it after model load
export_binary_weights()

# # After model loading in generate_text.py:
# print("Token 60 embedding:", model.embedding.weight[60][:5].tolist())

def generate_text(prompt, max_new_tokens=128, temperature=0.8, top_k=200):
    if prompt:
        # Split the prompt by spaces to get individual tokens
        prompt_tokens = prompt.strip().split()
        # Convert each token string to its index
        tokens = [stoi[token] for token in prompt_tokens]
    else:
        tokens = []
        
    device = next(model.parameters()).device  # Get device from model
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)[None, ...]
    
    # Generate one token at a time
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)
            logits = logits[:, -1, :] / temperature  # Get predictions for last token
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    
    # Get only the newly generated tokens (excluding prompt)
    prompt_length = len(prompt_tokens) if prompt else 0
    generated_text = ' '.join([itos[i] for i in tokens[0][prompt_length:].tolist()])
    
    # Split by spaces and get first 32 symbols
    symbols = generated_text.split()
    truncated_symbols = symbols[:32]
    
    # Rotate tokens until we don't start with '_' or '-'
    while truncated_symbols and (truncated_symbols[0].startswith('_') or truncated_symbols[0].startswith('-')):
        truncated_symbols = truncated_symbols[1:] + [truncated_symbols[0]]
        if not truncated_symbols:  # Safety check
            break
    
    result = ' '.join(truncated_symbols)
    return result


@app.route('/generate_text', methods=['POST'])
def generate_melody_endpoint():
    try:
        # try to parse json 
        data = request.get_json(force=True)
        prompt = data.get('prompt', "70 - 70 - 70 - _ _ ")  # default prompt if none provided

        generated_text = generate_text(prompt)
        print(f"Generated: {generated_text}")

        return jsonify({"generated_melody": generated_text})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


# cmd: curl -X POST -H "Content-Type: application/json" -d "{\"prompt\":\"70 - 70 - 70 - _ _ \"}" http://localhost:5000/generate_text