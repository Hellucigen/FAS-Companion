# FAS-Companion

FAS（Fascinator）的 **Companion Track**：让系统现在就能陪用户聊天、记录经历、
在游戏里互动。这一层的语言实现是**临时借用**的成熟方案（云端 LLM / 本地 Ollama、
OCR、ASR 等），全部经适配器接入，可按
[FAS-Companion ↔ Cognitive 接口指南](../FAS-Cognitive/docs/FAS_INTERFACE_GUIDE.md)
逐个替换为 FAS 自建机制——替换语言方案不需要修改认知层。

依赖方向单向：本仓 → **FAS-Cognitive**（同级目录，或 `FAS_COG_ROOT` 环境变量指向）。
认知机制的实验状态以 `FAS-Cognitive/docs/COGNITIVE_MODULE_STATUS.md` 为准。

## 仓库结构

```
app.py                    Flask 服务与回合主管道（/api/nlp；迁移目标形态见
                          FAS-Cognitive/fas/companion/pipeline.py）
index.html                单文件 UI（图谱观测 + 聊天）
nlp_processor.py          LLM 编排（解析/生成/抽取三类后端）
llm_provider.py           语言网关：所有云端调用物理收口于 _create() 一处
ollama_backend.py         可选本地后端（默认关）
prompt_templates.py       任务模板（含 FAS_IDENTITY 身份声明——全仓唯一 persona
                          硬编码点；无台词库，不预设说话风格）
api_guard.py              HTTP 安全（回环免token / Origin 核验 / 执行写闸门）
chat_log.py 之外：chat_log_replay.py  对话日志与事故恢复工具
ear/                      听觉（FunASR Paraformer STT + emotion2vec + YAMNet）
eye/                      屏幕感知（RapidOCR + mss + YOLO）
vision/                   物体感知（SAM2 轮廓 + DINOv2/CLIP 嵌入 + FAISS）
scripts/ tests/           陪伴侧实验脚本与测试
data/llm_config.example.json  复制为 data/llm_config.json 填入你的 api_key
```

## 快速开始

```bash
pip install -r requirements.txt
cp data/llm_config.example.json data/llm_config.json   # 填入 api_key（文件不入库）
python app.py            # http://127.0.0.1:5000
pytest tests/ -q
```

可选外部壳：**Charon / PersonalTerminal**（Go+Wails 桌面控制台）可托管本进程、
代理全部 `/api/*`，并经 Bridge(127.0.0.1:17734) 接收 FAS 推送
（utterance 事件/toast/日记）；FAS 侧通道实现已就位：
`FAS-Cognitive/fas/companion/charon_bridge.py`，
开启 `config.companion_charon_enabled` 即接入。

## 缺件登记（临时占位，待补充选型）

| 缺件 | 现状 |
|---|---|
| TTS 语音输出 | 未接入（候选：Charon 前端 Web Speech API / 本地 TTS） |
| STT 实时麦克风流 | ear/ 目前只支持手动喂音频文件 |
| 正式搜索 API | actions/web_search.py 为 HTML 抓取临时方案 |

## 隐私红线

`data/`（除 example）与 `logs/` 一律不入库：真实图谱、对话记录、内部状态、
LLM 密钥均只存本机。发布前经扫描闸门核查（个人标识词、密钥模式）。
