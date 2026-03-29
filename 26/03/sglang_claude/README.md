## Sglang

本来之前使用vllm部署的大模型，看到社区都在用sglang部署Qwen3.5。于是便采用sglang。

```sh
export SGLANG_DISABLE_CUDNN_CHECK=1

venv/bin/python -m sglang.launch_server \
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

在装包的过程中，也遇到了一些版本不一致的问题。参考了该 [sglang:blackwell ](https://docker.aityp.com/image/docker.io/lmsysorg/sglang:blackwell)docker镜像才解决了版本冲突。

踩坑经历：我使用sglang部署Qwen/Qwen3.5-35B-A3B-GPTQ-Int4失败了。折腾了好长时间都失败了，后来哪怕使用blackwell的sglang-docker镜像都部署不了。便放弃了该模型部署，不是因为显存空间不够。是MOE架构和GPTQ的包的问题。


当前可以直接浏览，本项目的 `pyproject.toml`文件，看到各个包的版本。

在 Claude Code 的设置文件，`~/.claude/settings.json` 。填写下述参数，即可完成配置。模型的名字无论写成什么(qwen3.5-91b)都能连接成功。

```sh
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

## LiteLLM

虽然Sglang部署的大模型，Claude Code可以直接用。但也看到很多人使用LiteLLM转发OpenAI格式的包使其支持 Anthropic。

官方文档：https://docs.litellm.com.cn/docs/tutorials/claude_code_max_subscription

pip install -U 'litellm[proxy]'

`lite_llm.yaml`:

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



```sh
litellm --config lite_llm.yaml --port 4000
```

修改 `~/.claude/settings.json` :

```
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.3.27:4000",
    "ANTHROPIC_AUTH_TOKEN": "sk-local",
    "ANTHROPIC_MODEL": "claude-3-5-sonnet-20241022",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-3-5-sonnet-20241022",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-3-5-sonnet-20241022",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-3-5-sonnet-20241022",
    "ANTHROPIC_REASONING_MODEL": "claude-3-5-sonnet-20241022"
  }
}
```

大模型的调用发送到4000端口。


## docker

docker: https://docker.aityp.com/image/docker.io/lmsysorg/sglang:blackwell

启动docker:
docker run -it --rm --gpus all lmsysorg/sglang:blackwell /bin/bash
docker run -it --rm --gpus all sglang-jie:latest /bin/bash


直接把镜像拉下来不能直接用，因为里面的transformers是旧版本，还不支持Qwen3.5，所以要把包更新一下。
docker镜像里面装包：
pip3 install --upgrade transformers --break-system-packages

保存镜像
docker commit xxx sglang-jie:latest