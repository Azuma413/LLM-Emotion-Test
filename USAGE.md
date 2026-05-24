# Usage

このリポジトリの実験は、各フェーズを CLI から順に実行する。重い学習は GPU 前提だが、`tabular_smoke` と小さいサンプル設定では CPU だけでデータ形式、交渉環境、評価出力を確認できる。

## 1. データ準備

```bash
uv run llm-emotion-test prepare-data --config configs/sft.yaml
```

WRIME を読み込み、SFT/蒸留用の JSONL を `datasets/processed/` に出力する。`data.max_samples` を小さくすると smoke test 用データを作れる。
変換済み JSONL の品質集計だけを再実行する場合は以下を使う。

```bash
uv run llm-emotion-test summarize-data --config configs/sft.yaml
```

集計は `data.summary_filename` に指定した JSON としても保存される。

## 2. 感情 SFT

```bash
uv run llm-emotion-test train-sft --config configs/sft.yaml
```

入力 soft prompt と出力 latent marker を含む SFT を実行し、`outputs/runs/<run_id>/checkpoints/` に checkpoint を保存する。

## 3. 自己蒸留

```bash
uv run llm-emotion-test distill --config configs/distill.yaml
```

教師モデルの感情指示付き応答を生成またはキャッシュから再利用し、学生モデルを教師出力へ合わせて学習する。教師生成のキャッシュ、蒸留 JSONL、学生用 JSONL は run directory に保存される。

## 4. GRPO / AT-GRPO

CPU smoke run:

```bash
uv run llm-emotion-test train-rl --config configs/rl_grpo.yaml
```

`configs/rl_grpo.yaml` の `rl_task.policy_backend` を `tabular_smoke` にすると、LLM 推論なしで交渉環境、rollout buffer、turn-wise grouped advantage、checkpoint 保存を確認できる。`llm` の場合は soft prompt モデルを使って各 agent turn の候補を生成し、報酬で相対比較する。
第三者回答器はデフォルトでは rule-based だが、`rl_task.third_party_backend: llm` にすると第三者 LLM が `<answer>1234</answer>` 形式の暫定回答を生成する。
LLM 版 AT-GRPO は checkpoint directory 内の `rl_state.pt` に optimizer state と episode index を保存し、`rl_task.resume_from_checkpoint` で再開できる。

主な出力:

- `outputs/runs/<run_id>/metrics.jsonl`
- `outputs/runs/<run_id>/rl_transcripts.jsonl`
- `outputs/runs/<run_id>/rollout_buffer.jsonl`
- `outputs/runs/<run_id>/checkpoints/`

## 5. 評価・観察・可視化

```bash
uv run llm-emotion-test evaluate --config configs/eval.yaml
```

固定 seed の交渉タスクに対して、以下の比較対象を同じ出力形式で評価する。

- `base_model`
- `sft_model`
- `distilled_model`
- `rl_model`
- `latent_fixed`
- `latent_random`
- `no_latent`

現時点の smoke 評価では、モデル checkpoint がない比較対象は rule-based surrogate と ablation agent で代替する。`evaluation.source_run_dir` に tabular GRPO run を指定し、`checkpoints/rl_checkpoint.json` が存在する場合、`rl_model` はその latent policy を読み込む。
`evaluation.sft_model_checkpoint_dir`、`evaluation.distilled_model_checkpoint_dir`、`evaluation.rl_model_checkpoint_dir` に soft prompt checkpoint が存在する場合、該当 variant は `LLMNegotiationAgent` として実モデルを読み込む。checkpoint がない場合は smoke 評価用 surrogate にフォールバックする。必ず実モデルを要求する場合は `evaluation.model_variant_backend: llm` を指定する。

評価は次を出力する。

- `evaluation_metrics.csv`: 平均 reward、合意率、Pareto efficiency、fairness、発話長、parser failure rate、latent entropy、task state と latent の相互情報量
- `evaluation_transcripts.jsonl`: variant ごとの transcript
- `model_comparison.jsonl`: 同一 task seed での variant 比較用レコード
- `latent_transition_heatmap.svg`
- `reward_curve.svg`
- `emotion_distribution.svg`
- `evaluation_report.md`: 数値サマリ、transcript sample、失敗例

## 6. 一括 smoke 確認

```bash
uv run pytest
uv run llm-emotion-test prepare-data --config configs/sft.yaml
uv run llm-emotion-test train-rl --config configs/rl_grpo.yaml
uv run llm-emotion-test evaluate --config configs/eval.yaml
```

## 7. 対話サンプル

```bash
uv run llm-emotion-test sample-dialogue --config configs/eval.yaml
```

ルールベース agent 同士で1 episodeを実行し、`outputs/runs/<run_id>/sample_dialogue.jsonl` に transcript を保存する。
