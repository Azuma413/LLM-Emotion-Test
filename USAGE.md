# Usage

このリポジトリの実験は、各フェーズを CLI から順に実行する。標準 config は WRIME 全件を使う GPU 実学習用で、`runtime.require_gpu: true` により CUDA GPU がない環境では失敗する。短時間の動作確認をしたい場合は、`data.max_samples`、`training.max_steps`、`rl_task.num_episodes`、`evaluation.num_tasks` を小さくする。CPU だけで交渉環境や評価出力を確認したい場合は、RL の `rl_task.policy_backend` を `tabular_smoke` に変更する。

## 1. データ準備

```bash
uv run llm-emotion-test prepare-data --config configs/sft.yaml
```

WRIME を読み込み、SFT/蒸留用の JSONL を `datasets/processed/` に出力する。`data.max_samples` を小さくすると smoke test 用データを作れる。
SFT JSONL は `soft_prompt(input_latent_id) + 文の前半` を入力し、`文の後半 + latent marker` を target にする continuation 形式で作られる。latent ID はテキスト prompt には書き込まれない。
変換済み JSONL の品質集計だけを再実行する場合は以下を使う。

```bash
uv run llm-emotion-test summarize-data --config configs/sft.yaml
```

集計は `data.summary_filename` に指定した JSON としても保存される。

## 2. 感情 SFT

```bash
uv run llm-emotion-test train-sft --config configs/sft.yaml
```

入力 soft prompt と WRIME 文 prefix から continuation と latent marker を生成する SFT を実行し、`outputs/runs/<run_id>/checkpoints/` に checkpoint を保存する。
標準設定では `outputs/runs/sft/checkpoints/final` が後続の自己蒸留の初期モデルになる。

## 3. 自己蒸留

```bash
uv run llm-emotion-test distill --config configs/distill.yaml
```

教師モデルの感情指示付き応答を生成またはキャッシュから再利用し、学生モデルを教師出力へ合わせて学習する。教師モデルは latent marker を生成せず、学生用 JSONL の target にだけルールベースで latent marker を付与する。教師生成のキャッシュ、蒸留 JSONL、学生用 JSONL は run directory に保存される。
`distillation.student_checkpoint_dir` に SFT の final checkpoint を指定すると、SFT 後のモデルを学生モデルとしてロードして蒸留する。標準設定では `outputs/runs/sft/checkpoints/final` から開始し、`outputs/runs/distill/checkpoints/final` を保存する。

## 4. GRPO / AT-GRPO

GPU LLM run:

```bash
uv run llm-emotion-test train-rl --config configs/rl_grpo.yaml
```

標準設定では `rl_task.policy_backend: llm` と `rl_task.resume_from_checkpoint: outputs/runs/distill/checkpoints/final` を使い、蒸留後の soft prompt モデルを AT-GRPO の初期 policy としてロードする。`configs/rl_grpo.yaml` の `rl_task.policy_backend` を `tabular_smoke` にすると、LLM 推論なしで交渉環境、rollout buffer、turn-wise grouped advantage、checkpoint 保存を確認できる。
交渉中の A/B agent は直接回答せず、相手に有用な制約共有だけを行う。第三者回答器はデフォルトでは rule-based だが、`rl_task.third_party_backend: llm` にすると第三者 LLM が `<answer>1234</answer>` 形式の暫定回答を生成する。
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

標準設定では `evaluation.model_variant_backend: llm` により、`evaluation.sft_model_checkpoint_dir`、`evaluation.distilled_model_checkpoint_dir`、`evaluation.rl_model_checkpoint_dir` の soft prompt checkpoint を実モデルとして読み込む。checkpoint がない場合は失敗する。短時間確認で surrogate へフォールバックしたい場合は `evaluation.model_variant_backend: auto` を指定する。

評価は次を出力する。

- `evaluation_metrics.csv`: 平均 reward、合意率、Pareto efficiency、fairness、発話長、parser failure rate、latent entropy、task state と latent の相互情報量
- `evaluation_transcripts.jsonl`: variant ごとの transcript
- `model_comparison.jsonl`: 同一 task seed での variant 比較用レコード
- `latent_transition_heatmap.svg`
- `reward_curve.svg`
- `emotion_distribution.svg`
- `evaluation_report.md`: 数値サマリ、transcript sample、失敗例

## 6. フル GPU パイプライン実行

```bash
uv run pytest
uv run llm-emotion-test prepare-data --config configs/sft.yaml
uv run llm-emotion-test train-sft --config configs/sft.yaml
uv run llm-emotion-test distill --config configs/distill.yaml
uv run llm-emotion-test train-rl --config configs/rl_grpo.yaml
uv run llm-emotion-test evaluate --config configs/eval.yaml
```

この一括実行は smoke backend ではなく、SFT、SFT checkpoint からの Distill、Distill checkpoint からの LLM AT-GRPO を順に実行する。標準 config は全データと `num_train_epochs` / `num_episodes` / `num_tasks` に基づいて実行するため、短時間確認では各 config のサンプル数やステップ数を明示的に小さくする。

## 7. 対話サンプル

```bash
uv run llm-emotion-test sample-dialogue --config configs/eval.yaml
```

ルールベース agent 同士で1 episodeを実行し、`outputs/runs/<run_id>/sample_dialogue.jsonl` に transcript を保存する。
