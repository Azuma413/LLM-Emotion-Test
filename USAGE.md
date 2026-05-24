# Usage

このリポジトリの実験は、各フェーズを CLI から順に実行する。重い学習は GPU 前提だが、`tabular_smoke` と小さいサンプル設定では CPU だけでデータ形式、交渉環境、評価出力を確認できる。

## 1. データ準備

```bash
uv run llm-emotion-test prepare-data --config configs/sft.yaml
```

WRIME を読み込み、SFT/蒸留用の JSONL を `datasets/processed/` に出力する。`data.max_samples` を小さくすると smoke test 用データを作れる。

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
