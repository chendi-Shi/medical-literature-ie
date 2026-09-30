# 文本分类实验协议 v2

本地冻结协议为 `thucnews-10k-v2`。protocol.json 的创建时间、数据 / 划分指纹及五个算法文件 SHA-256 均保存。它是本地可核验的实验记录，不是外部预注册。SVM 的相同配置、相同数据已完成训练产物被复用；所有测试评测在全部配方训练完成后统一进行。公开测试数据可查看，不宣称它是研究人员从未见过的盲测集。

研究问题：在相同样本、预训练小编码器和优化器预算下，R-Drop 与 FGM 是否改善十类新闻标题分类及合成扰动稳定性？收益是否足以抵偿计算成本？

## 固定数据 / 模型

每类 train 1,000，共 10,000；完整清洗验证 9,991；来源 test 10,000 全部保留。数据抽样种子 1729。神经随机种子 42/43/44，每个种子运行 CE / R-Drop / FGM / 联合四种配方。模型 BGE-small-zh-v1.5，预训练缓存修订 `7999e1d3359715c523056ef9478215996d62a620`，约 24M 参数；随机分类头，全参数微调。

2 epochs，batch 16，累积 2，max_length 48，AdamW LR 3e-5、weight_decay .01、梯度 clip 1、cosine、warmup ratio .06、FP16。每轮 625 个微批次，313 个更新窗口；正常总预算 626 次更新。记录 AMP 跳过更新数与截断计数。相同 seed 的初始化和 DataLoader 顺序相同，dropout 额外随机抽样不同；不保证跨设备逐位可复现。按 validation Macro-F1 保存最佳模型，相同分数保留更早 epoch。

两个稀疏模型使用字符 TF-IDF 1–3 gram、sublinear_tf、max_features 60,000；balanced Logistic（C=1、max_iter=500）或 LinearSVC（C=1、max_iter=5,000、dual=auto）。全部词表和参数仅拟合 train，未调超参。稀疏模型算法基本确定，只运行 seed 42，不伪造三种子标准差。

## 训练目标

普通 CE：一次带 dropout 前向得到交叉熵。

R-Drop：两次独立 dropout 前向，`Lclean = (CE1 + CE2)/2 + α × (KL(p||q) + KL(q||p))/2`，α=.5。KL 对类别求和，对样本取平均，两个分布均保留梯度。与采用不同 KL reduction 的实现不能直接比较 α。

FGM：`g = ∂Lclean/∂E`，对当前微批次计算输入嵌入矩阵梯度，`δ = ε g / ||g||₂`，ε=.5，范数针对全嵌入矩阵；零梯度不扰动。扰动下额外计算一次 CE，`Ltotal = Lclean + β CEadv`，β=.5。联合配方的扰动方向来自含 KL 的 Lclean，对抗项仍为单次 CE；不是双前向对抗 R-Drop。

用 autograd.grad 提取当前微批次方向，避免此前累积的 .grad 改变 δ。clean 与 adversarial 梯度按实际样本数累积，尾部窗口也正确平均；扰动结束或异常都恢复原嵌入。AMP 只对累计优化器梯度缩放，方向来自未缩放的 clean loss。额外 KL / 对抗 CE 分别记录。损失总值跨目标含义不同，不用总 loss 高低判断分类方法优劣。

## 选模 / 校准 / 统计

方法按三种子的 validation F1 平均值选择，稀疏方法按单次验证结果参与比较。全部训练完成后加载保存模型，测试不参与梯度和配方调整。

在 validation raw scores 上最小化 NLL 拟合单温度 T（.1–100），保存 calibration.json。正温度不改变 argmax；SVM 使用分类 margin，Logistic 使用 log probability，Transformer 使用 logits。校准与 checkpoint 选择共用验证集；温度和拒判阈值拟合不接触 test。拒判阈值为校准验证 confidence 的 20% 分位，测试报告实际覆盖率和接受样本准确率。ECE 用 15 个等宽置信度区间，NLL / Brier 同报。

主指标是完整 test Macro-F1；固定十类标签、每类 F1 均报告。神经方法为三种子均值±样本标准差。相对 CE 按相同训练 seed 配对，按 test 真实类别重采样样本行，400 次，seed 31415，95% percentile interval。相同索引用于三个固定模型种子。它只衡量固定模型下样本抽样不确定性，不涵盖训练种子总体不确定性；未做多重比较校正。来源 test 仍有少量重复，行重采样不是重复簇 bootstrap。

鲁棒性每类随机固定 200 条，共 2,000，seed 2027。每种方法共用原文、每六字符插入逗号、随机删 5%（至少一个）字符、尾部截断 30% 四组。报告相对同组原文的 F1 差值，不拿 2k 扰动分数与 10k 原文直接相减。删字 / 截断可能改变标签语义，沿用原标签，没有人工复标；这是合成压力测试。

## 资源与限制

相同 optimizer-step 不等于相同 FLOPs；报告实测单次训练时间、PyTorch peak allocated CUDA MiB，以及本机全测试打分吞吐（不含加载，不是生产服务基准）。CPU 稀疏与 GPU 神经吞吐不能直接当作同硬件算法效率结论。

本轮只验证一个公开新闻标题域、10k 训练预算和固定 2 epochs。没有超参搜索、学习曲线、训练预算曲线或自然分布外业务数据。公开数据可能进入预训练；R-Drop / FGM 组合是已有方法的适配与实验，不能据此宣称原创或普适优越。
