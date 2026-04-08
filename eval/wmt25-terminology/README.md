# WMT25 Terminology Shared Task Eval Script

## 评测
直接搬运自 WMT25 术语翻译任务 Track 1 的评测脚本，计算：
1. BLEU-4 翻译流畅度得分
2. chrF2++ 字符级翻译准确率
3. Acc 术语翻译准确率
4. Cons 术语一致性

## 数据文件

`eval/wmt25-terminology/ende.test.input.jsonl` 是 `datafiles/wmt25/track1/ende.proper.jsonl` 的提取测试版，由于 `datafiles/*` 目录下的数据文件庞大，不希望 Codex 直接读取或者 Git 跟踪，所以请以这个文件作为示例。

## 输出
参考 `eval/wmt25-terminology/ende.test.output.jsonl` 输出原文及对应译文的 `jsonl` 文件到 `datafiles/wmt25/track1/output` 目录中
## 运行脚本（NMT + LLM 后编辑）

新增脚本：`eval/wmt25-terminology/run_nmt_llm_eval.py`

- 输入：目录下一个或多个 `jsonl` 文件（每行至少包含 `en`，可选 `terms`）
- 输出：`jsonl` 文件，字段为 `en` 和你指定的目标语言（如 `de`）
- 评测链路：NMT 翻译 + LLM 后编辑
- 覆盖模块：TM 检索 + 提示词模块
- 不覆盖模块：术语匹配识别（直接使用输入 `terms` 作为已知正确术语）

运行方式：

```bash
eval $(poetry env activate)
python eval/wmt25-terminology/run_nmt_llm_eval.py \
  --input-file datafiles/wmt25/track1/enru.proper.jsonl \
  --target-lang ru \
  --domain-hint "information technology"
```

可选参数：

- `--output-dir`：输出目录（默认 `datafiles/wmt25/track1/output`）
- `--top-k`：TM 检索条数（默认 5）
- `--llm-temperature`：后编辑温度（默认 0.5）
- `--domain-hint`：领域提示词（默认 `信息技术`）

## 运行脚本（Track1 官方指标复现，单文件）

新增脚本：`eval/wmt25-terminology/evaluate_track1_local.py`

- 输入：某个系统输出 `jsonl`（每行至少包含 `en` 和 `de`）
- 参考：`datafiles/wmt25/track1/reference/full_data.ende.jsonl`
- 输出：与 `track1_score_de_dict.json` 同结构的 `json`
- 指标：`bleu4`、`chrf2++`、`proper/random term success rate`、`consistency_frequent/predefined`
- 一致性模块：已将原 OpenAI 调用替换为本系统 Google AI（`litellm`，默认 `gemini/gemini-2.0-flash`）

运行方式：

```bash
eval $(poetry env activate)
python eval/wmt25-terminology/evaluate_track1_local.py \
  --input-file datafiles/wmt25/track1/output/enru.proper.ru.output.jsonl \
  --target-lang ru \
  --mode proper \
  --system-name Glossa \
  --output-file eval/wmt25-terminology/result/glossa_track1_score_ru_dict.json \
  --consistency-model gemini/gemini-2.0-flash
```

说明：

- 当前实现默认只评测 `de` 目标语言。
- `term-consistency` 的 `awesome-align` 预训练模型改为按需懒加载，仅在 `de` 且发生过对齐时加载。

## 预加载模型

```
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
hf download stanfordnlp/stanza-es --local-dir stanza_models/es
hf download stanfordnlp/stanza-ru --local-dir stanza_models/ru
```


