# 中文信息抽取：开源对照与实现方案

目标是把中文原文转为带实体位置的关系结构，形成真实数据清洗、编码器微调、推理与评测闭环。网页默认进入实体与关系抽取，历史新闻分类实验保留在单独入口。二者的指标不能混用。

## 已阅读的开源依据

**PaddleNLP DuIE**：固定 v2.8.1 / 提交 `db99efd4dc99047922aae9842be66ab4538f93bf`，实际读取 [DuIE README](https://github.com/PaddlePaddle/PaddleNLP/blob/db99efd4dc99047922aae9842be66ab4538f93bf/examples/information_extraction/DuIE/README.md)。说明中采用 ERNIE token 多标签方案，处理重叠 SPO，支持复杂 object 多槽位组合及官方严格匹配评测。它的官方任务覆盖比本轮二元槽位评测更完整。迁移到当前 PyTorch 环境需要重新组织模型、数据和评测；不能把框架差异说成算法不足。

**作者 GPLinker**：实际读取 [duie_v1.py](https://github.com/bojone/GPLinker/blob/main/duie_v1.py)，读取日期 2026-10-01。实体指针识别 subject / object span，两个有向关系指针分别连接首 token 和尾 token；解码取两者交集。实体指针使用 RoPE 和上三角约束，关系首尾连接不应施加上三角限制。代码使用 bert4keras，并包含预训练模型、数据路径及首个表面字符串匹配。这里的 v1 数据和本轮 DuIE2 展开槽位不是同一基准。该链接未固定提交，不能声称复现特定论文结果。

**GPLinker_pytorch**：固定提交 `ff679b1e80a7adb0146d9ba66c9ddfe3107817a6`，实际读取 [README](https://github.com/JunnYu/GPLinker_pytorch/blob/ff679b1e80a7adb0146d9ba66c9ddfe3107817a6/README.md)。该版本说明通过 `run.py` 配置 epochs、maxlen、batch、预训练路径，并指出 datagenerator 的效率问题。参考它的 PyTorch 任务组织；没有把未成功读取的模型源码当成已审计内容，也没有复用 README 分数作为本项目结果。

## 本项目具体改进

选用本机可用的 BGE-small-zh 中文编码器，在 4GB 显存上实际训练，替代不可下载完整权重的 BERT-base。这是约 24M 参数的编码器微调，不是大语言模型训练。

实现 **带类型跨度的 GPLinker 扩展**：共享编码器输出经过 Efficient GlobalPointer，预测实体类型、关系首部和尾部连接；根据 schema 的主客体类型解码。保留嵌套跨度与多关系，避免普通单标签 BIO 的表达限制。它沿用已有算法，不声明原创或预先保证超过对照。

该实现不是作者源码的逐行复现：类型化实体头为 26 个，采用共享低维 Q/K 与直接来自编码器隐藏向量的各头首尾偏置；实体、关系首部、尾部三项损失分别对 batch 和 heads 平均再相加。作者 `duie_v1.py` 的 loss 先对 heads 求和再按 batch 平均，且实体头仅分 subject/object。头数和损失归约会改变各任务的相对权重，因此这里只评估当前明确记录的类型跨度变体，不能用它的分数代表 GPLinker 方法上限。

实现 **NER → 实体对分类流水线**：同一编码器联合优化实体跨度损失与候选实体对多标签分类损失；推理先解码实体，再对 schema 允许的实体对分类。训练实体对来自真实实体，正常推理只用预测实体，因此会有误差级联。额外输入真实实体的测试仅作诊断，不能作为可部署分数。

补充来源及数据指纹、固定不重叠划分、验证选 checkpoint 与阈值、两种子对照、完整测试、逐槽位指标和错误证据。独立训练冻结编码器模型，另用同一联合 checkpoint 去掉尾部交集做解码消融。按同组文本评测中性前缀压力，并给 seed42 主对照做文本配对 bootstrap。

推理服务输出实体 ID、类型、原文位置、关系槽位和主客体引用。长输入采用重叠窗口，合并重复跨度；前端以 Unicode code points 定位，支持 emoji。JSON 导出使用实际神经模型结果，不用规则或预设结果替代。

## 数据与评测边界

数据来自 [官方仓库的下载讨论](https://github.com/PaddlePaddle/PaddleNLP/issues/447)中的 DuIE2 压缩包。下载字节为 37,097,755，实测 SHA-256 为 `df223b18f3bd0fe19a61d18d7d59a22cd3f145c342c58470e93851bb4d1b40ed`，这是本地校验值，不是发布者签名。保留原始 `License.pdf` 与来源信息，数据和权重存于忽略的 workspace，不随 Git 分发。

源训练 171,293 条、dev 20,674 条。清洗保持原文不变；用于去重的规范化结果不替换原文。所有槽位必须精确对齐原文和 tokenizer 边界，无法完整对齐的整条记录隔离。字符位置按第一次精确出现推导，重复实体提及可能有语义位置歧义。训练排除与整个来源 dev 的规范化重叠，来源内去重。

本轮仅保留不超过 128 tokens 的原始记录，固定抽取训练 10,000、验证 2,000、测试 4,000 条。验证和测试是来源 dev 的不重叠子集，官方无公开标签的 test 不参与评测。训练原文含 18,243 个二元槽位监督；schema 为 48 种 predicate 展开的 55 种槽位、26 种实体类型。

实体 F1 是 **关系参与者的带类型字符跨度精确匹配**，不是独立完整 NER 基准。关系 F1 匹配带类型的主语表面字符串、predicate/slot、宾语表面字符串；不要求位置相同，不做别名宽松匹配。复杂 object 拆为多个二元槽位，**不等于 DuIE 官方完整多槽 SPO 指标**，不能与来源仓库分数直接比较。

主要限制是小编码器、10k 预算、长文本过滤、首次提及位置、有限 schema、没有完整 NER 标签、没有近重复检测和独立业务域测试。前缀压力不是自然分布外泛化；两种子统计不能证明普遍优越性。跨窗口关系与跨段落共指尚未建模，抽取分数未经校准，不代表事实核验。

## 架构与复现

`nlp_lab/ie/data.py` 下载与审计；`encoding.py` 字符/token 对齐和候选；`model.py` 神经抽取头与目标；`train.py` 微调与最佳 checkpoint；`predict.py` 解码和窗口；`metrics.py` 精确指标；`suite.py` 冻结协议和统计。API 与网页位于 `ie_api.py`、`web/ie.js`、`web/ie.css`。

固定训练为 6 epochs，batch 8、累积 4、LR 3e-5、AdamW weight decay .01、cosine、warmup .06、FP16，token 上限 160。主模型 seed42/43；冻结消融 seed42。每轮完整验证，零阈值关系 F1 选 checkpoint；随后在验证集网格 `0 / -.5 / .5` 选择共享实体/关系 logit 阈值。全部模型完成训练及阈值选择后打开测试集。按验证均值选架构，部署固定 seed42。

协议保存代码 SHA-256、数据 SHA-256、完整配方、阈值及测试政策；源码或配方变化时拒绝混入旧套件。每次产物记录实际环境、优化器更新、显存和完整 safetensors 权重。失败的单次训练从头开始，目前没有 optimizer 断点续训。

流程核验披露：正式套件之前，用 96 条训练、24 条验证的小规模单轮 checkpoint 做运行与重载自测，其中前 12 条测试文本也被用于指标管道核验。它们未用于正式选模、阈值或质量超参选择；AMP 初始 scale 修复只依据训练更新全部被跳过的问题。正式五个模型的完整测试统一在训练和验证选择之后进行，额外报告排除这 12 条后的 3,988 条关系指标。不能将本轮描述为测试文本从未被任何自测读取。

```powershell
python -m scripts.prepare_ie
python -m scripts.run_ie
python -m nlp_lab.ie.cli train --architecture joint --epochs 6
python -m nlp_lab.ie.cli extract --run RUN_ID --text "刘慈欣是《三体》的作者。"
python -m nlp_lab.ie.cli evaluate --run INDEPENDENT_RUN_ID --split test
python -m nlp_lab.cli serve --port 8778
```

准备模型缓存后可离线训练。本机 Python 为 `C:/anaconda/envs/dsproject/python.exe`；PowerShell 设置 `$env:PYTHONIOENCODING='utf-8'`，替换上面的 python。独立实验不进入主对照，冻结套件的评测产物禁止通过普通 CLI 覆写。

## 上下文与位置关系头扩展

手写推理例子发现：一个人物可能同时被判为电影的导演和主演，实体识别较好但角色区分不足。为此新增实体两侧和实体间上下文均值、主客体跨度、CLS 及有向距离嵌入，保持原流水线候选、负例、损失与优化器配置。不是关键词匹配，也不声明原创关系抽取算法。

扩展模型与运行入口独立于已冻结的 `nlp_lab/ie/`，通过进程内模型工厂复用同一训练器。两种子 6 轮配方于 2026-10-01T02:49:50Z 固定，原组首次正式测试在 02:51:21Z 开始；设计没有使用原组完整测试分数。扩展两次训练和验证阈值选择后，再评测相同 4,000 条与相同前缀压力样本。最终三种方法按两种子验证均值选择，固定 seed42 部署；负结果也保留。具体特征与代码 SHA 见 [扩展协议](IE_CONTEXT_PROTOCOL.json)。
