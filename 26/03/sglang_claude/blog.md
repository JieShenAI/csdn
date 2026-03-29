# 本地部署 Qwen3.5-9B：基于 Sglang + Claude Code 的完整实战教程

> 本文记录如何使用 sglang 在本地部署 Qwen3.5 模型，并配置 Claude Code 进行代码辅助开发。涵盖从命令启动、Docker 部署到常见问题排查的全过程。

---

## 一、为什么选择 Sglang？

之前使用 vllm 部署大模型时，看到社区开始转向使用 **sglang** 部署 Qwen3.5，于是我也跟进尝试。sglang 以其高效推理能力和灵活的配置选项，成为了本地部署大模型的优秀选择。

---

## 二、环境准备

### 硬件要求

- **GPU 显存**：9B 型号占用 28GB
- **系统环境**：Python 3.12+，CUDA 12.8

### 依赖包版本

可以直接查看项目的 `pyproject.toml` 文件，了解各个包的版本要求。本地安装时需要注意包版本的一致性。

---

## 三、本地部署 Sglang

### 1. 启动命令

使用 sglang 启动 Qwen3.5-9B 模型的完整命令：

```bash
export SGLANG_DISABLE_CUDNN_CHECK=1

.venv/bin/python -m sglang.launch_server \
--model-path Qwen/Qwen3.5-9B \
--served-model-name qwen3.5-9b \
--host 0.0.0.0 \
--port 8000 \
--tp-size 1 \
--mem-fraction-static 0.80 \
--kv-cache-dtype fp8_e4m3 \
--context-length 131072 \
--reasoning-parser qwen3 \
--tool-call-parser qwen3_coder \
--attention-backend triton \
--chunked-prefill-size 65535 \
--triton-attention-num-kv-splits 4 \
--max-running-requests 4 \
--enable-tokenizer-batch-encode \
--disable-radix-cache \
--enable-metrics
```

### 2. 参数详解

| 参数 | 说明 |
|------|------|
| `--model-path` | 模型路径 |
| `--served-model-name` | 服务模型名称 |
| `--tp-size` | Tensor Parallelism 大小 |
| `--mem-fraction-static` | 显存使用比例（0.85 表示使用 85% 显存） |
| `--context-length` | 上下文长度（131072 = 128K） |
| `--attention-backend` | 注意力后端（triton） |
| `--chunked-prefill-size` | Prefill 分块大小 |
| `--enable-metrics` | 启用 metrics 监控 |

### 3. 常见问题：版本冲突解决方案

在安装过程中可能会遇到版本不一致的问题。参考 [sglang:blackwell](https://docker.aityp.com/image/docker.io/lmsysorg/sglang:blackwell) Docker 镜像的依赖配置，可以有效解决版本冲突问题。

---

## 四、配置 Claude Code 使用本地模型

### 1. 环境变量配置

在 Claude Code 的设置文件 `~/.claude/settings.json` 中添加以下配置：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.3.27:8000",
    "ANTHROPIC_AUTH_TOKEN": "sk-local",
    "ANTHROPIC_MODEL": "qwen3.5-91b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3.5-91b",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3.5-91b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3.5-91b",
    "ANTHROPIC_REASONING_MODEL": "qwen3.5-91b"
  },
  "syntaxHighlightingDisabled": true
}
```

> 注意：模型名称可以随意填写（如 "qwen3.5-91b"），Claude Code 只通过 BASE_URL 和 TOKEN 来识别本地连接。

### 2. 效果验证

配置完成后，Claude Code 即可通过本地服务调用模型，无需访问外部 API。

---

## 五、使用 LiteLLM 转发 OpenAI 格式

虽然 Sglang 支持直接通过 Anthropic SDK 调用，但许多人更习惯使用 OpenAI 格式的接口。此时可以使用 **LiteLLM** 进行格式转换。

### 1. 安装 LiteLLM

```bash
pip install -U 'litellm[proxy]'
```

### 2. 配置文件 `lite_llm.yaml`

```yaml
model_list:
  - model_name: claude-3-5-sonnet-20241022
    litellm_params:
      model: openai/qwen3.5-9b
      api_base: http://127.0.0.1:8000/v1
      api_key: sk-local

litellm_settings:
  drop_params: false
```

### 3. 启动 LiteLLM 服务

```bash
litellm --config lite_llm.yaml --port 4000
```

### 4. 重新配置 Claude Code

修改 `~/.claude/settings.json`，将 BASE_URL 改为 LiteLLM 服务端口：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.3.27:4000",
    "ANTHROPIC_AUTH_TOKEN": "sk-local",
    "ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022"
  }
}
```

此时大模型请求会被转发到 4000 端口，并通过 OpenAI 格式与 Sglang 通信。

---

## 六、Docker 部署方案

### 1. 官方blackwell镜像

采用社区提供的 blackwell 镜像：

```bash
docker run -it --rm --gpus all lmsysorg/sglang:blackwell /bin/bash
```

### 2. 镜像构建步骤

**问题：** 直接拉取镜像后发现 Transformers 版本过旧，不支持 Qwen3.5。

**解决方法：** 在镜像内升级包版本

```bash
# 进入容器
docker run -it --rm --gpus all lmsysorg/sglang:blackwell /bin/bash

# 升级 transformers
pip3 install --upgrade transformers --break-system-packages

# 保存镜像
docker commit <container_id> sglang-jie:latest
```

---

## 七、踩坑记录：A3B-GPTQ-Int4 部署失败

### 问题描述

尝试部署 `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` 模型时失败，即使使用 blackwell Docker 镜像也无法成功。

**重要结论：失败原因不是显存不足。**

### 根本原因

1. **MoE 架构兼容性**：A3B（混合专家模型）架构对依赖包有特定要求
2. **GPTQ 量化包问题**：GPTQ 版本的依赖与 sglang 不兼容

### 推荐方案

- ✅ Qwen3.5-9B：稳定可用，推荐部署
- ❌ Qwen3.5-35B-A3B-GPTQ-Int4：不建议使用，等待官方修复

---

## 八、总结

通过 sglang + LiteLLM 的组合，实现了：
1. 本地高效部署 Qwen3.5-9B 模型
2. 直接通过 Anthropic SDK 调用
3. 通过 OpenAI 格式兼容 Others 的工具链
4. Docker 化部署便于环境复用和共享

**推荐配置优先级：**
1. 本地部署 sglang（适合开发调试）
2. LiteLLM 转发（兼容 OpenAI 工具）
3. Docker 镜像部署（适合长期运行）

---

*本文档记录于 2026-03-28，持续更新中...*
