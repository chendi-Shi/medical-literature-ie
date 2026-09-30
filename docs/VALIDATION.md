# 本机验证记录

日期：2026-09-30（Asia/Shanghai）。环境：Windows、Python 3.13.12 的 dsproject 环境、PyTorch 2.6.0+cu124、Transformers 5.9.0、PEFT 0.19.1；GPU 为 RTX 3050 Ti Laptop，4GB 显存。每次实验的精确 Python / 库版本在对应 run.json；requirements-tested.txt 记录本机依赖。

## 自动测试

`C:/anaconda/envs/dsproject/python.exe -m pytest -q`：**13 passed**。

覆盖规范化去重和冲突隔离、跨划分重叠拒绝、固定 seed 重现、文件篡改检测、未知标签/混合划分拒绝、held-out 独有字符未进入 TF-IDF 词表、模型重新加载、异数据版本比较拒绝、单训练锁、API 推理与本地边界、全参数/LoRA/分类头的训练与 checkpoint 重新加载。

神经网络测试使用本地创建的 tiny BERT，完全离线；每种方法对保存前后 validation log loss 做一致性检查。8 条 train、batch_size=3、accumulation=3 检查最后一个短 batch 和单次 optimizer step 的执行，另有三个真实缓存模型实验如下。

当前有依赖自身的弃用警告（Starlette/httpx、SciPy、tokenizer 与 SWIG）；不影响通过结果。已修复 FP32 softmax 舍入导致的 log_loss 误警告：先检查非负/概率和，再用 FP64 归一化。

## 真实训练与独立测试

数据版本：chinese-demo；指纹见 workspace/datasets/chinese-demo/manifest.json。90 条原创人工编写的合成示例，三个类别各 30 条。54 train / 18 validation / 18 test。**这些不是业务盲测或公开数据集成绩。**

- CPU TF-IDF 基线：`20260930-003528-aa920489`，validation Macro-F1 0.882051，test Macro-F1 0.885714，test accuracy 16/18；训练约 0.054 秒（不含进程启动）。
- BGE-small-zh 全参数微调：`20260930-003545-d175e2da`，4 epochs、batch=8、max_length=64、lr=5e-5；最佳 epoch=2；validation/test Macro-F1 均为 1.0；训练约 11.793 秒（含模型载入/保存，不含独立 test）；23,955,459 可训练参数。
- BGE-small-zh LoRA：`20260930-003806-1d6baa7f`，4 epochs、rank=8、query/value、lr=3e-4；最佳 epoch=4；validation/test Macro-F1 均为 1.0；训练约 17.475 秒；67,075 可训练参数 / 24,022,534 总参数。
- 另运行一次 `HF_HUB_OFFLINE=1` 的单 epoch LoRA：`20260930-004058-05b8622b`，验证 Macro-F1 0.604040；用于验证缓存载入和修复后 adapter 保存离线可用，不作为完整四轮实验的成绩。

基础模型快照：`7999e1d3359715c523056ef9478215996d62a620`。预训练编码器没有原有三分类头，因此出现 classifier 权重新增初始化提示是正常行为；分类头随后实际训练。adapter 加载包含保存的分类头，测试验证了重新载入后一致性。

对照 CLI / API 使用同一 SHA-256 数据版本与同一完整划分。没有重新选择 seed、修改测试文本或丢弃错误样本来提升成绩。此演示数据过小且主题词明显，不能证明任何模型在真实任务上的泛化优势。

## 网页验证

工作台在 http://127.0.0.1:8778 启动；浏览器可见真实实验、每轮曲线、验证指标与测试混淆矩阵。实际点击模型推理，输入“球队凭借最后一分钟的进球赢得比赛”，LoRA 模型输出体育和三类概率；单条概率未经校准。

实际网页训练提交验证发现学习率 input 的 min/step 组合引发原生 stepMismatch，已改为 step=any；重新点击后成功建立后台基线实验 `20260930-004217-849427e6`。训练状态、独立测试评测和错误样本均可在页面查看。

训练完成后不持续重绘实验详情，避免阅读错误样本或展开配置时被轮询折叠。训练/评测运行期间自动刷新状态。
