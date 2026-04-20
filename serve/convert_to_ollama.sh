#!/bin/bash
# Convert fine-tuned merged model to GGUF and import into Ollama
# Usage: bash serve/convert_to_ollama.sh

set -e
cd /home/johnlin/personal/fine-tune

MERGED_DIR="./code-review-model/merged"
GGUF_DIR="./code-review-model/gguf"
GGUF_FILE="${GGUF_DIR}/code-reviewer-q4_k_m.gguf"
CONVERT_SCRIPT="/home/johnlin/.local/share/virtualenvs/new_srampy-uKxbEByR/lib/python3.12/site-packages/bin/convert_hf_to_gguf.py"

echo "=== Step 1: Convert HF model to GGUF (FP16) ==="
mkdir -p "$GGUF_DIR"
python "$CONVERT_SCRIPT" "$MERGED_DIR" --outfile "${GGUF_DIR}/code-reviewer-f16.gguf" --outtype f16

echo ""
echo "=== Step 2: Quantize to Q4_K_M (~4.5GB) ==="
# Use llama-quantize if available, otherwise ollama can quantize on import
if command -v llama-quantize &>/dev/null; then
    llama-quantize "${GGUF_DIR}/code-reviewer-f16.gguf" "$GGUF_FILE" Q4_K_M
    rm "${GGUF_DIR}/code-reviewer-f16.gguf"
else
    echo "llama-quantize not found, using F16 directly (ollama will handle it)"
    GGUF_FILE="${GGUF_DIR}/code-reviewer-f16.gguf"
fi

echo ""
echo "=== Step 3: Create Ollama Modelfile ==="
cat > "${GGUF_DIR}/Modelfile" << 'EOF'
FROM ./code-reviewer-q4_k_m.gguf

SYSTEM """你是資深軟體工程師，專精程式碼審查與資安。請提供具體、有建設性的 code review。"""

PARAMETER temperature 0.1
PARAMETER num_predict 512
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
EOF

# Adjust Modelfile if we used f16
if [[ "$GGUF_FILE" == *"f16"* ]]; then
    sed -i "s/code-reviewer-q4_k_m.gguf/code-reviewer-f16.gguf/" "${GGUF_DIR}/Modelfile"
fi

echo ""
echo "=== Step 4: Import into Ollama ==="
cd "$GGUF_DIR"
ollama create code-reviewer -f Modelfile

echo ""
echo "=== Done! ==="
echo "Test with:  ollama run code-reviewer '請對以下程式碼做 code review: def get_user(uid): return db.execute(f\"SELECT * FROM users WHERE id = {uid}\")'"
echo "API:        curl http://localhost:11434/v1/chat/completions -d '{\"model\":\"code-reviewer\",\"messages\":[{\"role\":\"user\",\"content\":\"test\"}]}'"
