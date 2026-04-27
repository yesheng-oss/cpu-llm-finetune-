# CPU LLM Fine-tuning

一个轻量级的 LLM 微调项目，专为 CPU 设计。

## 环境要求

- Python 3.8+
- 8GB+ RAM
- 20GB+ 硬盘空间

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

在 `data/train.jsonl` 中准备训练数据，格式如下：

```json
{"instruction": "任务指令", "input": "输入文本", "output": "期望输出"}
```

### 2. 训练模型

```bash
python train.py --data_path ./data/train.jsonl --num_epochs 3
```

参数说明：
- `--model_name`: 模型名称，默认 Qwen/Qwen2-0.5B-Instruct
- `--data_path`: 训练数据路径
- `--output_dir`: 输出目录
- `--num_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--max_length`: 最大序列长度

### 3. 推理测试

单条指令测试：
```bash
python inference.py --model_path ./output --instruction "你是一个问答助手" --input_text "中国的首都是哪里？"
```

交互式对话：
```bash
python inference.py --model_path ./output --chat
```

## 项目结构

```
cpu-llm-finetune/
├── train.py          # 训练脚本
├── inference.py     # 推理脚本
├── requirements.txt
├── README.md
├── data/
│   └── train.jsonl  # 训练数据
└── output/         # 输出模型
```

## 支持的模型

可替换为其他小型模型：
- Qwen/Qwen2-0.5B-Instruct
- Qwen/Qwen2-1.8B-Instruct
- TinyLlama/TinyLlama-1.1B-Chat-v1.0
- openchat/openchat_3.5

## 注意事项

1. 首次运行会下载模型，可能需要较长时间
2. 根据实际硬件调整 batch_size 和 max_length
3. LoRA 参数可根据需求调整
