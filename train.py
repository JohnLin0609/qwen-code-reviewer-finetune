"""
Fine-tune Qwen2.5-Coder-7B for Code Review
===========================================
環境需求：
    pip install unsloth trl transformers datasets peft bitsandbytes accelerate

執行：
    accelerate launch --config_file accelerate_config.yaml train.py
"""

import json
import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from accelerate import Accelerator

# ── Accelerate（DDP-ready：單卡/多卡不需改 code）────────────────────────────
accelerator = Accelerator(mixed_precision="bf16")
print(f"裝置：{accelerator.device}, 程序數：{accelerator.num_processes}")

# ── 設定 ──────────────────────────────────────────────────────────────────

MODEL_NAME   = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_PATH    = "training_data_v7_final.jsonl"             # v7: 1,064 chat-format entries (391 fixed v6 + 673 synthetic)
OUTPUT_DIR   = "./code-review-model-v7"                   # separate output — keep previous models available
MAX_SEQ_LEN  = 2048

# LoRA 超參數
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

# 訓練超參數
EPOCHS       = 4       # bumped from 3: v7 has ~40% of v5 samples, more epochs compensate
BATCH_SIZE   = 1       # 8GB VRAM 用 1
GRAD_ACCUM   = 16      # 等效 batch = 1 * 16 = 16
LR           = 2e-4


# ── 載入模型 ──────────────────────────────────────────────────────────────

print("載入模型...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,                    # QLoRA，省一半顯存
    dtype=torch.bfloat16,                 # bf16 在 Ampere+ GPU 更快
)

# 加上 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",  # 省顯存
    random_state=42,
)

# 顯示可訓練參數數量
model.print_trainable_parameters()


# ── 準備資料 ──────────────────────────────────────────────────────────────

print("載入資料...")
raw = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw.append(json.loads(line))

# v7 已經是 chat 格式 (messages 欄位已包含 system/user/assistant)
def format_sample(sample):
    return {"text": tokenizer.apply_chat_template(sample["messages"], tokenize=False)}

print("格式化資料...")
formatted = [format_sample(s) for s in raw]
dataset   = Dataset.from_list(formatted)

# 切分訓練/驗證集（95% / 5%）
split   = dataset.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]

print(f"訓練集：{len(train_ds)} 筆")
print(f"驗證集：{len(eval_ds)} 筆")


# ── 訓練 ──────────────────────────────────────────────────────────────────

print("開始訓練...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        output_dir=OUTPUT_DIR,
        report_to="none",    # 改成 "wandb" 可以用 wandb 追蹤
        seed=42,
        # ── Accelerate / 記憶體優化 ──
        torch_compile=True,                # Ada GPU 支援，JIT 編譯加速 ~20%
        optim="paged_adamw_32bit",         # paged optimizer，自動 offload 到 CPU
        dataloader_pin_memory=True,        # 加速 CPU→GPU 傳輸（減少 PCIe 延遲）
        dataloader_num_workers=4,          # 平行載入資料，避免 GPU idle
        ddp_find_unused_parameters=False,  # DDP 優化：已知所有參數都會使用
    ),
)

trainer.train()


# ── 儲存 ──────────────────────────────────────────────────────────────────

print("儲存模型...")

# 只存 LoRA weights（小，幾十 MB）
model.save_pretrained(f"{OUTPUT_DIR}/lora")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora")
print(f"LoRA weights 儲存到 {OUTPUT_DIR}/lora")

# 合併並存成完整模型（大，約 15GB，可直接推理）
model.save_pretrained_merged(
    f"{OUTPUT_DIR}/merged",
    tokenizer,
    save_method="merged_16bit",
)
print(f"完整模型儲存到 {OUTPUT_DIR}/merged")

print("✅ 訓練完成！")
