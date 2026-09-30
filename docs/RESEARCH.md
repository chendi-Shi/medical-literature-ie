# GitHub 代码研究与方案选择

检索与核验日期：2026-09-30。先使用用户指定的 GitHub 连接器搜索仓库，再读取实际源码和固定 commit；没有只根据 README 列功能。优秀项目的选择依据是与任务的相关性、可扩展架构及成熟模型生态；不把 GitHub star 数量当作效果证据。

任务边界：用户选择模型训练项目，要求数据清洗、微调、推理与评测。未指定具体业务数据，因此首版采用中文单标签主题分类；内置数据只用来验证流程。对“大模型后训练”的需求，本机另有 posttrain-studio，当前项目建立独立分类任务与实验目录。

## Transformers：主干接口与训练示例

固定版本：`f339035b986aaf719bc6f5ea92342f73c498cb0e`，2026-09-29 更新。[官方仓库](https://github.com/huggingface/transformers)，[实际分类脚本](https://github.com/huggingface/transformers/blob/f339035b986aaf719bc6f5ea92342f73c498cb0e/examples/pytorch/text-classification/run_classification.py)。

框架：模型、tokenizer、数据预处理与 Trainer 分离；Auto* 接口适配不同预训练架构；训练示例提供任务参数、数据加载、训练、评估与模型保存。

功能证据：脚本 630–636 行为默认指标选择，单标签默认 accuracy；638 行开始 compute_metrics；664–672 行构建 Trainer；685 行保存模型；约 593–600 行在缺少 validation 时允许回退 test 并发出警告。

针对本项目的不足与风险：默认 accuracy 对不均衡标签不够充分；validation 缺失时回退 test 容易让不了解流程的使用者用测试集选模；所读示例没有专门的跨划分规范化重复审计与冲突隔离流程。这些结论仅针对所读脚本；不表示整个 Transformers 库不支持相关扩展。

采用：AutoTokenizer、AutoModelForSequenceClassification、save_pretrained。补充：validation 必须存在，数据指纹与泄漏检查、Macro-F1 选模、逐条预测与同数据版本对照。小规模分类项目采用可读的 PyTorch 训练循环，让梯度累积与保存规则清楚可查。大规模分布式任务应采用成熟 Trainer/Accelerate，而非扩充本版循环。

## Sentence Transformers：训练目标与评测解耦

固定版本：`4a3b5cd6ec718e421f57e824a41ed3fd99595df6`，2026-09-21 更新。[官方仓库](https://github.com/huggingface/sentence-transformers)，[实际训练器](https://github.com/huggingface/sentence-transformers/blob/4a3b5cd6ec718e421f57e824a41ed3fd99595df6/sentence_transformers/sentence_transformer/trainer.py)。旧 UKPLab 仓库地址已跳转到 huggingface 组织；源码路径也改为 sentence_transformer 子目录，以实际读取位置为准。

框架：SentenceTransformerTrainer 继承 BaseTrainer，配置模型、数据、loss 与 evaluator。第 73–77 行说明 evaluator 的指标通常比 eval loss 更适合衡量任务结果；124–129 行接受 loss 和 evaluator。

功能：围绕句向量及检索训练提供可替换目标与任务评测接口，适合将模型质量从训练 loss 中分离评估。

本项目的取舍：原生句向量或相似度目标与单标签分类的监督目标不同，不能拿下降的对比损失当作分类效果。这里借鉴 loss 与业务指标分离的设计，分类用交叉熵，选模用 Macro-F1。没有重写句向量检索系统。

## LlamaFactory：统一数据转换与微调配置

固定版本：`ce9dc9e072f80fa3abe0989d4ab90da25f083438`，2026-09-28 更新。[官方仓库](https://github.com/hiyouga/LlamaFactory)，[实际数据加载器](https://github.com/hiyouga/LlamaFactory/blob/ce9dc9e072f80fa3abe0989d4ab90da25f083438/src/llamafactory/data/loader.py)。

框架：统一加载、格式转换、不同阶段 processor 与数据划分；loader 第 25 行引入 split_dataset，第 315 行实际调用。官方 README 描述多种生成模型高效微调能力，包括 LoRA 和后训练流程。

功能与优势：生成式微调方法与配置覆盖广，适合指令、对话等后训练任务。

本项目的取舍：对三类短文本分类，使用生成模型会引入回答解析、生成耗时与更大的资源开销；这是一项任务适配判断，非代码缺陷或同设备基准结论。借鉴统一数据入口与配置化方法，当前采用更小的中文编码器和 PEFT 分类 LoRA。本版没有复现 LlamaFactory 的完整训练平台。

## 更适合本项目的方案

1. **先锁定数据契约**：规范化、冲突标签隔离、跨划分重叠拒绝；保留来源行号和审计记录。
2. **建立可追溯实验**：固定 seed、数据 SHA-256、配置与版本；每次训练独立目录。
3. **先做可解释基线**：字符 TF-IDF + Logistic Regression，成本低、可以揭露数据与标签问题。
4. **增量引入预训练模型**：全参数、仅分类头、LoRA 三条路径共用分类数据；4GB 设备优先小编码器与短序列，具体 batch 是否可用由实际运行验证。
5. **验证集选模与独立测试**：训练不使用 test 更新参数或选择 checkpoint；只有同一划分和标签映射才能比较。
6. **暴露失败证据**：每类指标、混淆矩阵、错误样本、原始概率、截断计数；不隐藏未完成的实验或合成数据来源。

“更好”的验收是闭环可运行、数据边界明确、结果可追踪，而非宣称创造了优于成熟开源库的新算法。真实业务上的优劣需要用户的真实训练/测试数据、多随机种子和统一资源预算来验证。
