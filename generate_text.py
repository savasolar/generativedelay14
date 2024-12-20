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


def export_for_cpp(model, checkpoint, output_path):
    # Get the model to CPU for numpy conversion
    model.to('cpu')
    
    # Extract and save all weights and token mappings
    weights = {
        'token_embedding': model.embedding.weight.data.numpy(),
        'position_embedding': model.position_embedding.weight.data.numpy(),
        'attention_qkv': model.attention.in_proj_weight.data.numpy(),
        'attention_bias': model.attention.in_proj_bias.data.numpy(),
        'lstm_ih': model.lstm.weight_ih_l0.data.numpy(),
        'lstm_hh': model.lstm.weight_hh_l0.data.numpy(),
        'lstm_bias': (model.lstm.bias_ih_l0.data + model.lstm.bias_hh_l0.data).numpy(),
        'output': model.output.weight.data.numpy(),
        'output_bias': model.output.bias.data.numpy(),
        'token_to_idx': checkpoint['token2idx'],
        'idx_to_token': checkpoint['idx2token']
    }
    np.savez(output_path, **weights)


# Then call it right after your model loading:
export_for_cpp(model, checkpoint, 'model_weights.npz')



# Update tokenizer loading:
stoi = checkpoint['token2idx']
itos = checkpoint['idx2token']

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