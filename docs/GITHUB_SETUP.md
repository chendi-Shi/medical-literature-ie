# 安装、复现与仓库范围

推荐 Python 3.13。创建独立环境并在仓库根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[training,test,report]"
python -m pytest -q
python scripts/verify_medical_snapshots.py
python -m nlp_lab.cli serve --port 8778
```

Linux 可用 `source .venv/bin/activate`。训练需要合适的 PyTorch CUDA 安装；自动测试使用 CPU 构造的小编码器，不下载预训练模型、不进行完整医学训练。工作台默认只监听本机。

新 clone **不附带已训练权重、原始语料、论文 XML 或 SQLite**。空工作区显示未训练，不会把仓库中的历史分数伪装成已部署的模型。已有实际模型评测见 [MEDICAL_RESULTS.md](MEDICAL_RESULTS.md)；报告 JSON 的哈希可以离线核验。请先查看各原始数据来源协议，下载脚本的 SHA 是本项目核验值，并非发布方签名。

完整医学复现：

```powershell
python scripts/cache_medical_encoder.py
python scripts/fetch_medical_data.py
python scripts/run_medical.py
python scripts/import_medical_papers.py
python scripts/export_medical.py
python -m nlp_lab.cli serve --port 8778
```

首次缓存模型和下载数据需要网络；训练和模型推理读取本地缓存。训练 runner 只在未完成的工作区执行对应阶段，已有协议源代码或数据指纹改变会拒绝覆盖。原始训练数据需自行获取，不在 GitHub 再分发。本机曾用 4GB RTX 3050 Ti、FP16 和累积更新完成训练，其他设备的耗时和结果不保证相同。

在工作台选择文献、提交抽取、等待完成。新版本支持本句终点 / 数值候选：点击“填入核验表单”后检查原文，再提交核验。它不自动批准候选，也不从其他句子猜组别。可切换抽取版本查看原预测和各版本的修订。全部审计导出包含逐条修订历史；已核验导出会剔除驳回实体并修剪引用。

仓库仅包含本项目代码、文档、测试、实际实验的轻量报告与少量带来源的核验片段。它不是医疗专家验证的临床决策系统。暂未为原始第三方数据和模型声明新的再分发许可；需要复用这些资源时以各来源协议为准。
