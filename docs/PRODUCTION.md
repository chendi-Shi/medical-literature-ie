# 部署准备与当前边界

v0.8 增加了一个受保护的单账号部署模式，适合在 TLS 反向代理后进行小规模内部试用。默认模式仍只绑定本机，方便开发。这个项目还没有达到经临床专家验证的生产系统标准：医学关系模型在 CMeIE 保留集上的 F1 为 28.83%，低于原模型 37.64%；疗效、组别和不良事件记录仍由规则生成候选。部署保护不能补足模型质量或临床验证。

## 生产模式的保护

- 服务只绑定 `127.0.0.1`，并限制允许的 Host 与唯一 HTTPS origin。TLS 在同机 Nginx 终止，应用只信任来自本机代理的转发头。不能把 8778 端口直接暴露到公网。
- 生产启动要求账号、至少 20 字符的密码、至少 32 字节的会话密钥、允许的 Host 列表和 HTTPS origin；缺少或配置错误时拒绝启动。登录使用 12 小时签名会话、HttpOnly、Secure、SameSite=Strict cookie 和有界的登录失败限速。生产 API 禁用交互式 OpenAPI 文档，并返回禁止缓存与基础安全响应头。
- 生产核验历史采用登录账号作为核验人，忽略请求正文中自行填写的核验人；开发模式保留原来的单机行为。
- `/health/live` 和 `/health/ready` 不包含文档或模型详情。Readiness 检查已训练模型文件及数据库连通性；它不执行推理，也不代表医学性能达标。
- 应用入口逐块计数并将请求体限制在 2 MB，覆盖缺失 `Content-Length` 的分块请求；Nginx 同时设置请求大小上限。医疗记录使用 SQLite WAL。`PRAGMA user_version` 记录数据库结构版本。生产示例将访问日志关闭，避免把带搜索词的 URL 写入反向代理日志。

这套登录只有一个共享账号，不支持 SSO、MFA、多租户、角色权限或多人身份生命周期管理。登录限速是单进程状态。若组织需要多人审核或公网访问，应接入组织 IdP、反向代理的强身份认证和集中审计，再完成独立安全评审；不要向团队分发同一个账号。

## 运行条件

先在受限环境取得有再使用权的数据与权重，运行医学训练和评测，将生成的工作区放在仅服务账号可读的目录。新 clone 不含数据库、文献、模型权重或术语资料；readiness 在这些文件缺失时会返回不可用。应用以单进程运行，每次只处理一个文献抽取任务；进程重启会将未完成任务标记为 `interrupted`，需由操作人员重新提交。不要增加 Uvicorn worker 数量，也不要在多个实例间共享 SQLite 工作区。

数据库里保存全文、抽取和人工修订。把工作区与备份放在访问权限严格的**加密磁盘**上，以专用操作系统账号运行。SQLite 备份不会加密文件。定期备份还须包含模型、配置和所需术语表，并在隔离环境演练恢复。依据组织的文献许可、个人信息保护要求、保留期限和删除流程设置访问、留存与审计策略。

## 保护配置

通过系统服务的受保护 secret manager 或仅 root/服务账号可读的环境注入设置这些值，切勿写入仓库或反向代理配置文件：

```text
MEDICAL_APP_USER=operator
MEDICAL_APP_PASSWORD=<至少20字符的高强度唯一密码>
MEDICAL_SESSION_SECRET=<至少32字节的随机秘密>
MEDICAL_ALLOWED_HOSTS=medical.example.org
MEDICAL_PUBLIC_ORIGIN=https://medical.example.org
```

确保用户、反向代理 `server_name`、Host 列表和 HTTPS origin 完全一致。轮换账号密码时也同时轮换会话密钥，以撤销已有会话。应用启动时会验证配置。

## HTTPS 反向代理

`deploy/nginx-medical.conf.example` 是一个最小参考。将占位域名和证书路径替换为组织自己的值，并设置防火墙只允许代理连接应用端口。示例包含 TLS、请求大小限制、登录限速、并发上限和不记录搜索参数的访问日志。证书申请、更新、外网防护和主机补丁由部署方负责。

将 `deploy/medical-upstream-headers.conf.example` 安装为 `/etc/nginx/snippets/medical-upstream-headers.conf`。把限速区和 `log_format` 放进 Nginx `http {}`，再把两个 `server` 块放进站点配置并校验、重载 Nginx。默认示例把应用和代理放在同一台主机；如果拆分主机，必须另行配置防火墙、可信代理来源和转发头，不能直接沿用 `127.0.0.1` 的信任设置。

服务示例位于 `deploy/medical-workbench.service.example`。编辑其中的路径、用户和受保护环境文件位置，再由系统服务管理器启动：

```sh
sudo install -D -m 0640 -o root -g medical /path/to/environment \
  /etc/medical-workbench/environment
sudo install -m 0644 deploy/medical-workbench.service.example \
  /etc/systemd/system/medical-workbench.service
```

环境文件只放入本机 secret manager 导出的上述变量。先创建 `medical` 系统用户及工作区，再按此模板实际安装。该 systemd 示例面向 Linux；Windows 部署需使用等效的服务管理器和 ACL。

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now medical-workbench
```

部署后先检查 `GET /health/live`、`GET /health/ready` 和登录页，再用已授权的测试文献核对上传、全文偏移、修订历史、导出与代理日志。不要在日志、终端记录、工单或聊天中记录密码、session cookie、全文或带查询词的 URL。

## 数据备份

使用 SQLite 在线备份 API，快照会在发布前通过 `PRAGMA quick_check`，命令拒绝覆盖已存在的文件：

```sh
nlp-medical --workspace /srv/medical/workspace backup \
  --output /mnt/encrypted-backups/literature-2026-10-07.sqlite3
```

目标必须处在加密的独立备份卷上；定期检查备份清单、读权限和可恢复性。更新程序或数据库结构前先建立并验证备份。当前没有自动备份调度或自动恢复功能。

## 放行门槛

当前默认关系模型仍为原两阶段 checkpoint，因为两组 GPLinker 候选都没有超过基线。不要启用低于门槛的候选来满足部署要求。进入真实临床流程前，至少还要用代表目标人群和文本来源的专家标注数据进行盲测和错误复核，覆盖关系、否定、引用研究、组别与不良事件；报告跨种子波动、域外性能、置信度和人工工作量，并由临床、隐私及安全负责人确定可接受阈值。新增文献类型、语言或人群都需要重新评估。
