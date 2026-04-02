# Fly.io 部署指南

## 前置条件

- [Fly CLI](https://fly.io/docs/flyctl/install/) 已安装并登录 (`fly auth login`)
- Gemini API Key ([获取](https://aistudio.google.com/apikey))
- Cloudflare R2 凭证（endpoint URL、access key、secret key、bucket name）

## 部署步骤

### 1. 创建 Fly App

```bash
cd sentrysearch
fly launch --no-deploy
```

提示选择时：
- App name: `sentrysearch`（或自定义）
- Region: 选离你最近的（如 `nrt` 东京、`hkg` 香港、`sjc` 硅谷）
- 不需要 Redis/Memcached

### 2. 创建 PostgreSQL 数据库

```bash
fly postgres create --name sentrysearch-db --region sjc
```

选择配置：
- Development (单节点，最便宜): `1 shared CPU, 256MB RAM, 1GB disk`
- 生产建议: `1 shared CPU, 1GB RAM, 10GB disk`

将数据库挂载到 app：

```bash
fly postgres attach sentrysearch-db
```

这会自动设置 `DATABASE_URL` secret。

### 3. 启用 pgvector 扩展

连接到数据库：

```bash
fly postgres connect -a sentrysearch-db
```

执行：

```sql
CREATE EXTENSION IF NOT EXISTS vector;
\q
```

> pgvector 在 Fly Postgres 镜像中已预装，只需 CREATE EXTENSION。

### 4. 设置 Secrets

```bash
fly secrets set \
  GEMINI_API_KEY=your-gemini-api-key \
  R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com \
  R2_ACCESS_KEY_ID=your-r2-access-key \
  R2_SECRET_ACCESS_KEY=your-r2-secret-key \
  R2_BUCKET=creative-assets
```

验证 secrets 已设置：

```bash
fly secrets list
```

应显示：

```
NAME                   DIGEST           CREATED AT
DATABASE_URL           xxxxxxxx         ...
GEMINI_API_KEY         xxxxxxxx         ...
R2_ACCESS_KEY_ID       xxxxxxxx         ...
R2_BUCKET              xxxxxxxx         ...
R2_ENDPOINT_URL        xxxxxxxx         ...
R2_SECRET_ACCESS_KEY   xxxxxxxx         ...
```

### 5. 部署

```bash
fly deploy
```

首次部署会：
1. 构建 Docker 镜像（Python 3.12 + ffmpeg + uv）
2. 推送到 Fly 镜像仓库
3. 启动 machine
4. 应用启动时自动建表（chunks、assets、evaluations）

### 6. 验证

```bash
# 健康检查
curl https://sentrysearch.fly.dev/health
# 应返回: {"status":"ok"}

# 查看日志
fly logs

# 查看 app 状态
fly status
```

## 常用运维命令

```bash
# 查看日志（实时）
fly logs

# SSH 进入容器
fly ssh console

# 查看 machine 状态
fly status

# 扩缩容
fly scale count 1        # 单实例
fly scale memory 1024    # 调整内存到 1GB
fly scale vm shared-cpu-2x  # 2 shared CPU

# 重启
fly apps restart

# 连接数据库
fly postgres connect -a sentrysearch-db

# 更新 secrets
fly secrets set GEMINI_API_KEY=new-key

# 查看部署历史
fly releases
```

## 配置调整

### fly.toml

```toml
app = "sentrysearch"
primary_region = "sjc"

[build]

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = "stop"      # 空闲时自动停机（省钱）
  auto_start_machines = true       # 有请求时自动启动
  min_machines_running = 0         # 允许停到 0（省钱）

[[vm]]
  memory = "1gb"
  cpu_kind = "shared"
  cpus = 2
```

### 关闭自动停机（如需常驻）

如果 asset-tracker worker 需要频繁调用 sentrysearch，建议关闭自动停机以避免冷启动延迟：

```toml
[http_service]
  auto_stop_machines = "off"
  min_machines_running = 1
```

### 调整请求超时

视频索引可能耗时较长（下载 + 分片 + 嵌入）。Fly 默认 HTTP 超时 60 秒。如需更长：

```toml
[http_service]
  [http_service.concurrency]
    type = "requests"
    hard_limit = 25
    soft_limit = 20
```

也可以在 Fly proxy 层设置：

```bash
fly proxy 8080:8080  # 本地代理，绕过 HTTP 超时限制
```

## 预估成本

| 组件 | 规格 | 月费（约） |
|------|------|-----------|
| Machine | shared-cpu-2x, 1GB RAM | ~$7 |
| Postgres | 1 shared CPU, 256MB, 1GB disk | ~$3.5 |
| 带宽 | 出站流量 | 前 100GB 免费 |
| **总计** | | **~$10.5/月** |

> 开启 auto_stop_machines 后，空闲时不计费，实际成本可能更低。

## 故障排查

**部署失败：Docker build error**
```bash
fly deploy --verbose  # 查看详细构建日志
```

**应用启动失败：DATABASE_URL not set**
```bash
fly secrets list  # 确认 DATABASE_URL 存在
fly postgres attach sentrysearch-db  # 重新挂载
```

**pgvector CREATE EXTENSION 失败**
```bash
fly postgres connect -a sentrysearch-db
# 确认连接的是正确的数据库
\dx  # 列出已安装扩展
```

**API 返回 500: Gemini API key error**
```bash
fly secrets set GEMINI_API_KEY=correct-key
fly apps restart
```

**请求超时（索引大文件）**
- 考虑拆分为小批次索引
- 或使用 `fly proxy` 绕过 HTTP 超时
