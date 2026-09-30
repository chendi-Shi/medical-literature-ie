# 数据说明：THUCNews 标题子集 / 固定 10k 预算

来源为 [649453932/Chinese-Text-Classification-Pytorch 固定提交](https://github.com/649453932/Chinese-Text-Classification-Pytorch/tree/6cb26819af7b646275aff8a4693676f2849e67f6) 中 THUCNews/data 的 class、train、dev、test。它是第三方抽取的十类新闻标题，不能称为官方 THUCNews 全文数据。财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

原始文件 train 180,000 / dev 10,000 / test 10,000。每个下载文件校验 Git blob SHA-1，再保存 SHA-256 与原始 URL，下载缓存位于 workspace/sources/thucnews-titles。源码 LICENSE 同时保存，源码许可证不自动证明新闻文本的商业再分发权；原始文本不提交本项目 Git。

按 NFKC、格式 / 控制字符去除、空白合并规范化，额外 casefold 作为精确重复键。来源 test 全部行及标签保留；与 test 重叠的 9 个 dev 行隔离，dev 剩余 9,991。train 去除与 dev / test 重叠行，冲突组全部隔离，同标签重复只保留首条，清洗后 178,707。审计共 1,302 条排除记录（9 dev、1,293 train），包含来源行号和原因。

seed 1729 从每个类别的 clean train 随机抽取 1,000 条，train 共 10,000；168,707 个清洗合格标题因预算未使用，记作 budget_omitted，不算清洗失败。最终数据版本共有 29,991 行。每条保留 source_file / source_line，audit.json 与 manifest.json 可追溯。

来源 validation 内有 5 个重复组（均无冲突）；test 有 10 个重复组，其中 1 组同时标为股票 / 财经。为保持来源测试口径，保留并在 manifest 披露；这意味着不能达到同文本多标签组的全部正确预测，指标也受重复行权重影响。没有删除困难样本或重标测试标签。数据加载器只允许这一明确声明的 heldout 内部重复策略，跨划分仍禁止重复。

数据 SHA256：`f7c0af5c473788ac0165e359d3dce54930d1dced8fc8ca3a48f14c9278d8f5d5`。三个划分均独立记录指纹，模型训练与评测前重新核验。一般用户上传的显式数据仍直接拒绝跨划分重叠，冲突标注隔离；本数据的保留策略是基准适配入口明确声明的例外。

局限：仅标题且单一新闻域；历史数据的采集年代与原始抽样过程未独立核验；精确去重无法消除语义近重复；预训练语料可能包含公开标题。没有自然分布外业务集，也没有人工复核删字 / 截断后的标签。不能把本轮随机 10k 训练子集与原仓库 180k 训练的分数直接比较。
