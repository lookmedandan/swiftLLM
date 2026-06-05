import pandas as pd
import json

# 1. 读取 Parquet
df = pd.read_parquet('../ultrafeedback-dpo/train.parquet')

# 2. 采样 1% (建议在转换时直接采样，减少 IO)
df_sample = df.sample(frac=0.01, random_state=42).reset_index(drop=True)

print(f"正在处理 {len(df_sample)} 条采样数据...")

# 3. 转换并清洗
dpo_data = []
for _, row in df_sample.iterrows():
    # 基础清洗：确保字段存在且不为空
    prompt = str(row['instruction']) if pd.notna(row['instruction']) else None
    chosen = str(row['chosen_response']) if pd.notna(row['chosen_response']) else None
    rejected = str(row['rejected_response']) if pd.notna(row['rejected_response']) else None

    if prompt and chosen and rejected:
        dpo_data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen}
            ],
            "rejected_response": rejected
        })

# 4. 保存为 JSONL
output_path = '../ultrafeedback-dpo/train_swift_dpo_1pct.jsonl'
with open(output_path, 'w', encoding='utf-8') as f:
    for item in dpo_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"转换完成！原始采样: {len(df_sample)}, 有效数据: {len(dpo_data)}")
print(f"保存路径: {output_path}")