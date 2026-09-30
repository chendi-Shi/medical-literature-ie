# 开源代码分析与方案 v2

用户目标是实际文本分类研究：真实多类别数据、强基线、模型改进、鲁棒性与消融。首版合成三类示例只能证明工程流程；v2 把研究问题、预算和统计证据加入相同训练项目。

## 参考代码

[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/6cb26819af7b646275aff8a4693676f2849e67f6)：固定提交 `6cb26819af7b646275aff8a4693676f2849e67f6`，实际阅读 README、train_eval.py、utils.py。仓库覆盖 TextCNN / TextRNN 等中文分类模型，训练使用 CE、验证 loss 选 checkpoint，最终测试有分类报告与混淆矩阵。utils 的词表只由训练文件构建，这是正确的隔离，不将其误判为数据泄漏。

对本任务的缺口：所读训练入口没有多种子汇总、校准、扰动压力评测或配对统计；数据规范化、冲突及跨划分重复也需要额外审计。仓库 README 的 200k 数据是第三方 THUCNews 十类标题子集；其 180k 训练结果不能与本轮 10k 对比。以上仅针对所读代码及本任务，非对所有分支 / 衍生项目的判断。

[R-Drop 官方仓库](https://github.com/dropreg/R-Drop/tree/9dcc8302eb9b0f112c38230864bbee807bbe1aa7) 与 [论文](https://arxiv.org/abs/2106.14448)：官方实现将 dropout 输出一致性用于多类任务，含 Hugging Face BERT 分类示例和修改过的训练代码。可借鉴两次前向、平均 CE、双向 KL；直接搬入旧版改动的整个 transformers 源码会增加维护成本。本项目在独立 PyTorch 损失模块实现核心目标，共用现代模型加载、保存与评测，不声称复现所有论文任务及成绩。

[Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725)：文本嵌入上的梯度扰动提供局部稳定性训练依据。本项目实现一次 FGM 梯度方向扰动及额外有标签 CE，不实现 VAT 或论文的全部半监督流程。扰动大小和目标 reduction 必须明确，累积梯度与 AMP 也需要验证。

首版对 [Transformers 分类脚本](https://github.com/huggingface/transformers/blob/f339035b986aaf719bc6f5ea92342f73c498cb0e/examples/pytorch/text-classification/run_classification.py)、Sentence Transformers 和 LlamaFactory 的工程调查保存在 [历史研究](RESEARCH.md)。任务已聚焦编码器分类，生成式指令微调框架的丰富能力并不等于这项分类研究需要引入全部生成栈。

## 更适合本目标的方案

数据层固定来源提交、Git blob 校验和 SHA-256，清洗保留行号；清理 train 与 heldout 重叠，不修改来源 test 标签。预算统一到每类 1,000，适应本机 4GB GPU。后续可扩大预算，但必须建立新协议，不能把不同预算混入本轮结论。

模型层用字符 TF-IDF Logistic / SVM 强基线检验预训练编码器是否值得；再做 CE / R-Drop / FGM / 联合 2×2 因子实验，每个神经配方 3 个种子。损失模块独立，计算时间和显存并列报告，防止只堆方法名字和单次分数。

实验层训练前固定数据、超参、种子、测试政策和扰动。验证选 checkpoint，完整测试统一评测；单温度校准、拒判阈值和配对 bootstrap 提供置信度与改善幅度证据。网页每项结果都来自本地模型产物，不使用预设曲线。

交付是可复现实验软件与当前预算的实际结论，不承诺方法组合一定胜出。实际结果见 [RESULTS_V2.md](RESULTS_V2.md)；协议与限制见 [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md)。
