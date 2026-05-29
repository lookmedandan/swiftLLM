import json
import os

# 修复 tokenizer 配置
model_path = "/models/qwen1.5b_muice_dpo_final"
tokenizer_config = os.path.join(model_path, "tokenizer_config.json")

if os.path.exists(tokenizer_config):
    with open(tokenizer_config, 'r') as f:
        config = json.load(f)

    # 修复 extra_special_tokens 格式
    if "extra_special_tokens" in config and isinstance(config["extra_special_tokens"], list):
        config["extra_special_tokens"] = {}

    with open(tokenizer_config, 'w') as f:
        json.dump(config, f, indent=2)

    print("修复完成")
else:
    print("文件不存在，跳过")