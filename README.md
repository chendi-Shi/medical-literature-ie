# 中文信息抽取 NLP Lab · v0.3

从中文原文识别实体、关系，并输出带原文位置的结构化 JSON。使用真实 DuIE2 数据、中文预训练编码器微调、NER → 实体对关系分类对照，以及带类型跨度的 GPLinker 联合抽取。包括数据清洗、独立训练、推理、完整测试、逐槽位错误、两种子统计、消融和前缀压力评测。

主入口是信息抽取。之前的文本分类研究保留在“历史文本分类实验”，详见 [分类项目文档](docs/CLASSIFICATION_README.md)；分类分数不能作为信息抽取效果。

## 使用

双击 `start.cmd`，或在项目目录执行 `./start.ps1`，打开 http://127.0.0.1:8778 。

- **实体与关系抽取**：选择真实训练 checkpoint，输入中文，查看实体类型、原文高亮、关系箭头与 JSON；可新增独立微调实验。
- **研究与评测**：实际训练进度、两种子均值及标准差、配对区间、实体误差级联诊断、冻结和解码消融。点击模型查看逐轮 F1、55 槽位指标与测试错误。
- **方法与数据**：已阅读的 GitHub 代码版本、算法取舍、固定划分、清洗统计和数据指纹。

输出是神经模型预测，不是事实核验。抽取分数未经校准；有限 schema 不覆盖任意自定义关系。字符位置按 Unicode code points、左闭右开表示，前端兼容 emoji；重叠窗口不保证跨窗口关系。

## 真实数据与方法

源数据训练 171,293、dev 20,674 条。整条记录的主客体和所有 object 槽位必须完整对齐原文及 tokenizer。原文不做破坏字符位置的规范化；规范化仅用于去重。训练排除与整个来源 dev 重叠的文本，来源内部精确去重。

本轮固定 **10,000 训练 / 2,000 验证 / 4,000 测试**，只保留不超过 128 tokens 的记录。训练包含 **18,243 个二元关系槽位**，覆盖 **55 槽位、26 实体类型**。验证与测试是来源 dev 的不重叠子集，官方无公开标签的 test 未用于评测。

三种方法共享 `BAAI/bge-small-zh-v1.5` 中文编码器（4 层、约 24M 参数），6 epochs、batch 8 × 累积 4、LR 3e-5、cosine、warmup .06、FP16；主模型 seed42/43，另训练冻结编码器 seed42。每轮验证选 checkpoint，验证网格选择 logit 阈值，再统一打开测试集。部署按验证均值选架构，固定 seed42。

实体指标只覆盖关系参与者的派生字符跨度，不是独立完整 NER 基准。关系按带类型主客体表面字符串及 predicate/slot 精确匹配，复杂 object 展开为二元槽位；**不等于官方 DuIE 完整多槽 SPO 指标**。首次精确匹配存在重复提及位置歧义。当前预算和指标不能直接对比来源仓库分数。

代码阅读范围和方案见 [IE_RESEARCH.md](docs/IE_RESEARCH.md)，实际结果及可写入简历的准确表述见 [IE_RESULTS.md](docs/IE_RESULTS.md)。数据协议 `License.pdf` 随下载保留；数据和模型不随 Git 分发。

## 复现与独立实验

```powershell
python -m scripts.prepare_ie
python -m scripts.run_ie
python -m scripts.run_ie_context
python -m scripts.export_ie
python -m scripts.analyze_ie
python -m nlp_lab.ie_commands train --architecture joint --epochs 6 --seed 42
python -m nlp_lab.ie_commands extract --run RUN_ID --text "刘慈欣是《三体》的作者。"
python -m nlp_lab.ie_commands extract --run CONTEXT_RUN_ID --text "刘慈欣是《三体》的作者。"
python -m nlp_lab.ie_commands evaluate --run INDEPENDENT_RUN_ID --split test
python -m nlp_lab.ie_import own.jsonl --schema own-schema.json --name own-v1
python -m nlp_lab.ie_commands train --dataset own-v1 --architecture joint --epochs 6
python -m pytest -q
python -m nlp_lab.cli serve --port 8778
```

本机 Python 为 `C:/anaconda/envs/dsproject/python.exe`，RTX 3050 Ti Laptop 4GB。PowerShell 设置 `$env:PYTHONIOENCODING='utf-8'`，并将上面的 python 替换为 `& 'C:/anaconda/envs/dsproject/python.exe'`。新机器安装 `pip install -e ".[training,test,report]"`，按设备选择 CUDA PyTorch，先准备模型缓存；训练不执行远端自定义代码。

本轮使用模型修订 `7999e1d3359715c523056ef9478215996d62a620`。换机器复现时应准备该修订；模型默认读取本地缓存。实际模型修订、环境、参数量、AMP 更新、时间与显存记录在每次 run，报告附 checkpoint SHA-256。

研究在训练前冻结代码和数据指纹、配方与评测政策；改动核心源码会拒绝混入旧套件。失败训练从头重训，已完成且匹配协议的实验可复用，目前没有 optimizer 断点续训。普通 CLI 禁止覆写研究评测产物。

早期纯流程自测曾使用前 12 条测试文本核验小规模 checkpoint 重载，未用于正式选模或阈值；原组五个正式模型在该组训练和验证选择后统一评测；新增上下文两种子在两次扩展训练和验证选择后统一评测，配方在原组测试开始前固定。分组报告另外核验排除这 12 条后的 3,988 条关系指标，详见 [流程披露](docs/IE_RESEARCH.md) 与 [难度分组](docs/IE_SUBGROUPS.md)。

## 目录

自有数据可在网页“方法与数据”导入。JSONL 每行包含 `text` 和 `spo_list`，每个 SPO 指定 subject、subject_type、predicate、object 槽位字典与 object_type 槽位类型字典。schema JSON 数组例如 `[{"predicate":"任职","subject_type":"人物","object_type":{"@value":"机构"}}]`。至少 20 条独立文本；无法对齐、过长与标注冲突记录隔离，跨划分重叠拒绝。可显式指定全部 split，否则 seed42 按文本 80/10/10 划分；训练必须覆盖全部 schema 槽位。

显式空 `spo_list` 表示已标注无已知关系文本，可作为负例；没有标注的原文不能当负例。实体仍只监督关系参与者，重复提及按首次精确匹配推导。自有实验单独保存，可在网页异步评测完整测试集；不进入 DuIE2 冻结对照。CPU 独立 CLI 训练指定 `--precision fp32`。

`nlp_lab/ie/` 提供数据、编码、模型、训练、推理、评测与套件；`ie_api.py` 和 `web/ie.*` 是本地单用户工作台，服务只监听 127.0.0.1。

`workspace/ie/datasets/` 保存划分、来源行与清洗审计；`workspace/ie/runs/` 保存完整 safetensors、tokenizer、配置、逐轮历史和错误；`workspace/ie/research/` 保存冻结协议与汇总。可分享的数值和代码指纹快照在 `docs/ie-results/`，不附原始数据正文。

当前未完成全量长文本训练、完整独立 NER 标注、跨窗口关系、共指消解、真实业务域外部测试、近重复去重、多 GPU、断点续训和生产部署。已有方法的工程实现与可复验对照，不声明原创算法或榜单成绩。

## 关系上下文与位置增强

针对手写推理例子中的“导演 / 主演”语义角色混淆，新增 `nlp_lab/ie_context.py`。关系表示由主语、宾语、实体间文本、各实体左右 4-token 窗口、CLS 共 8 个特征组成，另学习 16 维有向距离桶嵌入。局部均值排除 CLS、SEP 与 padding，无关键词规则。

扩展协议在原组测试开始前固定（[快照](docs/IE_CONTEXT_PROTOCOL.json)），另训练 seed42/43，沿用原训练器、数据、候选负例、损失和 6-epoch 预算。它通过进程内明确的模型工厂替换复用冻结训练器，原核心源码不改动。扩展两次训练及验证阈值选择后才做扩展测试；最终按三种方法的两种子验证均值选模型，不按测试结果挑种子。

网页可独立训练上下文增强模型，推理及评测自动加载正确的抽取头。统一入口 `nlp-ie` 或 `python -m nlp_lab.ie_commands` 自动适配三种抽取头，使用 `--architecture context_pipeline` 独立训练增强模型；网页也支持此流程。冻结核心的旧 CLI 留作版本证据，使用新公共入口。两个研究套件均会校验冻结源码与数据指纹；要改模型，应建立新协议版本。
