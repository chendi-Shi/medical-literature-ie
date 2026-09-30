# NLP Classification Research Lab · v0.2

真实中文多类别文本分类研究项目：数据清洗、稀疏强基线、预训练编码器微调、模型改进、三种随机种子、因子消融、扰动评测和校准推理。网页默认展示研究结果，原 90 条合成样本仅保留作流程自测。

本轮使用第三方 THUCNews **新闻标题**十类子集：从 180,000 条来源训练数据中清洗得到 178,707 条，固定每类抽取 1,000 条训练；验证 9,991 条、测试 10,000 条。完整来源测试行及标签保留，验证 / 测试内部重复与冲突披露。它不是官方 THUCNews 全文基准。

## 使用

双击 `start.cmd` 或执行 `./start.ps1`，访问 http://127.0.0.1:8778 。本机使用 `C:/anaconda/envs/dsproject/python.exe`，RTX 3050 Ti Laptop 4GB；无需升级已有环境。

- 研究总览：14 次实验进度、三种子均值与标准差、消融置信区间、鲁棒性、计算成本。
- 实验工作台：查看保存 checkpoint 的混淆矩阵、逐条错误、loss 与验证曲线；运行经过温度校准的推理；新增独立实验。
- 数据与审计：导入自己的 CSV / JSONL，核验划分、标签、来源与 SHA-256。
- 方法与依据：固定 GitHub 代码版本、R-Drop / FGM 来源及实现边界。

## 复现实验

在项目目录执行；本机把 `python` 替换为 `& 'C:/anaconda/envs/dsproject/python.exe'`。

```powershell
# 下载固定 Git 提交的数据文件，校验 Git blob SHA 和 SHA-256，生成清洗审计
python -m scripts.prepare_research
# 运行 2 个稀疏基线 + 4 种神经配方 × 3 seeds；已有匹配产物会复用
python -m scripts.run_research
# 可选图表依赖：pip install -e ".[report]"；核验并导出 PNG / PDF 与逐种子证据
python -m scripts.export_research
# 测试与本地服务
python -m pytest -q
python -m nlp_lab.cli serve --port 8778
```

模型默认读取本地缓存，预训练初始化为 `BAAI/bge-small-zh-v1.5`。本轮模型修订、实际依赖、设备与参数量记录在每个 run 中；不执行远端自定义代码。新机器需要先准备该模型权重，或对单次训练显式传入 `--allow-download`。

研究配方在 `workspace/research/thucnews-10k-v2/protocol.json` 训练前冻结；算法源码改变后，套件拒绝混入旧协议。所有训练完成后才统一评测测试集。浏览器与普通 CLI 拒绝覆盖研究套件的评测产物。失败后套件可重跑，复用完成实验；未完成的单个训练会从头重训，本版没有 checkpoint 断点续训。

## 本轮方法

TF-IDF 字符 1–3 gram、最多 60,000 特征，分别配 balanced Logistic Regression 与 Linear SVM，词表 / IDF / 分类器只拟合 train。神经模型是 BGE-small-zh 初始化的分类编码器，不使用检索损失：全参数微调，CE、R-Drop、FGM、R-Drop+FGM，seeds 42/43/44。固定 2 epochs、batch 16、累积 2、LR 3e-5、cosine、warmup 6%、48 tokens、FP16。R-Drop α=.5，FGM ε=.5，对抗损失权重=.5。

所有方法使用相同数据；神经方法使用相同优化器步数预算，但额外正则化的计算量不同。报告实际训练时间、PyTorch 峰值分配显存与单次打分吞吐。选 checkpoint 和方法只看验证 Macro-F1；不按最佳测试种子汇报。

评测包含完整测试集 Macro-F1 / accuracy / 每类指标 / 混淆矩阵 / 错误样本；三种子均值和样本标准差；相对 CE 的按类配对 bootstrap 条件 95% 区间；同组 2,000 条文本的标点、删字、截断压力测试。单温度在验证集拟合，报告 NLL、Brier、15-bin ECE 和拒判阈值的测试覆盖率。校准与选模共用 validation，并非独立校准集。

实测数值与结论见 [研究结果](docs/RESULTS_V2.md)，固定实验细节见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)，来源与代码分析见 [开源研究](docs/RESEARCH_V2.md)。组合已有方法，不声明原创算法。

## 独立实验 / 自有数据

```powershell
python -m nlp_lab.cli prepare data/my-data.jsonl --name my-data-v1 --provenance "来源与采样说明"
python -m nlp_lab.cli train --dataset my-data-v1 --backend baseline --baseline-model linear_svm
python -m nlp_lab.cli train --dataset my-data-v1 --backend transformer --regularization rdrop_fgm --epochs 2 --batch-size 16 --accumulation 2 --max-length 48 --learning-rate 0.00003 --schedule cosine --warmup-ratio 0.06 --mixed-precision fp16
python -m nlp_lab.cli evaluate --run RUN_ID --split test
python -m nlp_lab.cli predict --run RUN_ID --text "球队进入决赛"
python -m nlp_lab.cli compare RUN_1 RUN_2 --split validation
```

JSONL 行格式为 `{"text":"球队进入决赛","label":"体育","split":"train"}`，CSV 为同名列。UTF-8；split 可省略，自动按类固定种子划分。一般用户数据的跨划分重叠直接拒绝，同文本冲突标签隔离、同划分重复保留首条。公开基准专门按保留来源测试集的策略处理，详见 [数据说明](docs/DATA_CARD.md)。两种策略不能混为一谈。

独立实验支持全参数、LoRA 和冻结编码器；本版 FGM 必须使用全参数模型的可训练嵌入，不能与 LoRA / head 组合。FP16 / BF16 需要支持的 CUDA 设备，CPU 使用 FP32。普通训练不自动拟合温度；研究套件会保存 calibration.json，推理重载后自动使用。

## 架构与产物

`benchmark.py` 下载和清洗真实基准；`data.py` 版本与完整性；`algorithms.py` 正则化；`training.py` 训练和推理；`research.py` 冻结实验 / 校准 / 统计；`api.py` 与 `web/` 为本地工作台。

`workspace/datasets/` 保存数据、审计和来源行号；`workspace/runs/` 保存配置、历史、模型、评测、原始打分和扰动错误；`workspace/research/` 保存协议、固定扰动、状态和汇总。数据 / 权重不提交 Git；可分享的数值报告与协议快照在 `docs/results/`。查看 [架构](docs/ARCHITECTURE.md) 和 [本机验证记录](docs/VALIDATION_V2.md)。

新机器：`pip install -e ".[training,test]"`；CUDA PyTorch 安装源按设备选择。`requirements-tested.txt` 是本机实际版本。服务为本地单用户工具，只绑定 127.0.0.1。当前未实现近重复清理、真实业务外部测试、多 GPU、断点续训、NER、生成式 SFT/DPO 或生产服务部署。此轮数据规模和预算不能与来源仓库 180k 训练结果直接比较。
