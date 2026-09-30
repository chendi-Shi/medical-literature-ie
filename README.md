# NLP Training Lab

中文文本分类的完整训练项目：**数据清洗 → 固定划分 → 传统基线 → 预训练模型微调 → 独立评测 → 推理**。附本地网页工作台、CLI、代码研究依据和测试。

首版任务是单标签分类，内置体育 / 科技 / 财经三类中文主题示例。它是学习与实验原型；“更好的方案”指数据治理、实验可追溯性和适合本机资源的任务设计，不表示模型效果已超过研究的开源框架。

## 在这台电脑上使用

双击 `start.cmd`，或运行 `./start.ps1`，打开 **http://127.0.0.1:8778**。当前复用 `C:/anaconda/envs/dsproject/python.exe`，不升级现有 Python 环境。默认读取本地模型缓存，不联网下载。

- 「实验工作台」选择数据版本，启动 CPU 基线或 Transformer 微调。
- 「数据与审计」导入 CSV / JSONL，查看分布、划分与 SHA-256 指纹。
- 点击实验查看训练 loss、验证 Macro-F1、配置、混淆矩阵与错误样本。
- 训练完成后单独点击「评测测试集」，再在「模型推理」输入文本。
- 全参数、LoRA、仅分类头三种方法共用同一数据和评测流程。默认 BGE-small-zh，是预训练中文编码器初始化的分类模型；此项目没有训练句向量检索目标。

内置 **90 条原创人工编写的合成示例**，每类 30 条，固定 seed=42 划分为 54 train / 18 validation / 18 test。分数只能用于验证流程，不能代表真实业务效果。网页中的每个实验会显示来源说明。

## CLI

以下命令在项目目录执行，本机请将 `python` 替换为 `& 'C:/anaconda/envs/dsproject/python.exe'`：

```powershell
# 完整 CPU 自检，包括训练、独立测试集评测和模型保存
python -m nlp_lab.cli demo
# 真正从预训练中文编码器微调，并重新加载最佳模型做测试
python -m nlp_lab.cli demo --transformer --method full
python -m nlp_lab.cli demo --transformer --method lora
# 接入你的数据
python -m nlp_lab.cli prepare data/my-data.jsonl --name my-data-v1 --provenance "来源、许可与采样说明"
python -m nlp_lab.cli train --dataset my-data-v1 --backend baseline
python -m nlp_lab.cli train --dataset my-data-v1 --backend transformer --method lora --learning-rate 0.0003 --batch-size 4 --accumulation 2
# 使用 train 返回的实际实验 ID 替换下面的 RUN_ID
python -m nlp_lab.cli evaluate --run RUN_ID --split test
python -m nlp_lab.cli predict --run RUN_ID --text "球队进入决赛"
python -m nlp_lab.cli compare BASELINE_RUN TRANSFORMER_RUN --split validation
python -m pytest -q
```

所有命令支持前置 `--workspace PATH` 指定隔离工作区。API 文档在 `/docs`。

## 数据格式和清洗规则

JSONL 每行是 `{"text":"球队进入决赛","label":"体育"}`。CSV 第一行是 `text,label`。统一要求 UTF-8；label 为字符串。可另加 `split`，只允许 `train` / `validation` / `test`。一旦使用显式划分，所有有效记录都必须指定；三份划分必须非空，评测标签必须在 train 中出现。

未提供 split 时，每类至少 5 条独立文本；先按类别和规范化文本排序，再用 seed 洗牌。validation/test 各为 `max(1, floor(每类数量×0.2))`，其余进入 train。更严谨的按用户、文档、时间划分应在导入前完成，并提供 split。

清洗使用 NFKC、移除格式/控制字符、合并空白；去重键额外 casefold。空样本、超长文本、非法标签类型被排除并记录来源行号；同一文本冲突标注全部隔离；同划分重复保留首条；跨划分重叠直接拒绝，避免静默改动测试集。**精确去重无法保证消除语义近重复。**

数据目录含 `manifest.json`、`audit.json` 和三个 JSONL 划分。每次加载核验文件指纹和标签，训练分配后或评测前发生篡改会拒绝。完整性检查会读取所有划分，但测试数据不参与拟合、梯度更新或选模。

## 训练与评测

- **基线**：字符 1–3 gram TF-IDF + balanced Logistic Regression；词表、IDF 和分类器只拟合 train。验证集不调整这条基线的参数。
- **微调**：Hugging Face 分类模型 + PyTorch 自定义训练循环，单标签交叉熵、AdamW、固定学习率、梯度裁剪；按实际累积窗口样本数平均梯度，正确处理不足 batch 和尾部窗口。当前使用 FP32。
- **LoRA**：PEFT `SEQ_CLS`，默认 query/value、rank=8、alpha=16、dropout=0.05。分类头与 adapter 一起保存；基础模型需在缓存中可用。其他模型的模块名不同，应通过 `--lora-targets` 指定。
- **仅分类头**：冻结预训练编码器；保留随机初始化分类头训练，适合低显存试验。
- **选模**：每轮评测 validation，按最高 Macro-F1 保存模型；没有提升达到 patience 后早停。相同分数保留更早 checkpoint。
- **指标**：accuracy、Macro-F1、weighted F1、log loss、每类 precision/recall/F1 和混淆矩阵；固定全标签集合，报告缺失支持类别。
- **解释**：逐条真实标签、预测、概率与错误样本；输出概率未经校准。
- **对照**：CLI 拒绝混用数据版本、标签映射或评测指纹，网页单个实验不跨数据集排名。

不依赖第三方追踪服务。每个实验独立保存配置、来源、数据指纹、随机种子、实际版本、设备、参数量、截断数量、历史曲线、最佳 epoch 和模型。`workspace/.training.lock` 防止同一工作区并行训练；进程异常被杀后可能留下锁，确认记录的 PID 已退出后才移除锁并重跑。

## 新机器安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[training,test]"
python -m nlp_lab.cli train --dataset my-data-v1 --backend transformer --model BAAI/bge-small-zh-v1.5 --allow-download
```

CUDA 对应的 PyTorch wheel 根据设备安装。`requirements-tested.txt` 记录本机验证版本；`pyproject.toml` 约束已验证的 Transformers / PEFT 小版本。下载模型须显式 `--allow-download`；默认不执行远端自定义代码。离线先跑基线无需模型权重。

## 文件入口与边界

`nlp_lab/data.py` 为清洗与划分；`training.py` 为训练、加载和对照；`metrics.py` 为指标；`api.py` / `web/index.html` 为本地工作台；`tests/` 覆盖泄漏拒绝、指纹、标签映射、模型重新加载、对照限制与 API。

研究细节见 [docs/RESEARCH.md](docs/RESEARCH.md)，架构与路线见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，本机运行记录见 [docs/VALIDATION.md](docs/VALIDATION.md)。

本版尚不包含 NER、生成式 SFT/DPO、多机训练、断点续训、超参搜索、近重复检测、概率校准、多人认证与生产部署。模型加载只用于本项目生成的可信本地模型产物；服务仅绑定回环地址。
