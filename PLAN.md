# LLM Emotion Test 実装計画

## 前提

この計画は、人間が手作業で実装するための TODO ではなく、Codex が段階的にリポジトリを拡張していくための実装計画である。README.md の研究構想を、再現可能な Python プロジェクトとして実装することを目的にする。

最終的な到達点は、以下の一連のパイプラインを CLI から実行できる状態である。

1. 感情ラベル付きデータセットを準備する。
2. Qwen 系モデルに対して、感情 soft prompt と再帰入力用 soft prompt を使った SFT を行う。
3. 教師モデルの感情指示付き応答から、学生モデルへ自己蒸留する。
4. 複数 LLM エージェントの交渉タスク環境を構築し、GRPO で学習する。
5. 学習された感情表現が、タスク達成や対話構造にどう寄与したかを評価・可視化する。

## 実装方針

- 既存の `src/llm_emotion_test` 配下を正式な実装先とする。
- モデル・データセット・出力先は設定ファイルで切り替えられるようにする。
- 最初から大規模学習だけを狙わず、小さいモデル・小さいデータ・短いエピソードで動く smoke test を必ず用意する。
- 学習は GPU 前提だが、データ処理、設定検証、トークン整形、評価集計は CPU だけでもテストできるように分離する。
- soft prompt は通常の自然言語トークンではなく、モデル埋め込み層に挿入する learnable embedding として扱う。
- 再帰入力用 soft prompt は、出力テキスト末尾に専用マーカーとして表現し、学習時・推論時に latent ID または latent vector へ復元できる設計にする。

## フェーズ 0: プロジェクト基盤整備

### 目的

研究コードを拡張しやすい Python パッケージへ整える。

### 作業内容

- `pyproject.toml` に必要な依存関係を追加する。なお，ライブラリは直接書き込むのではなく， uv add経由で追加する事．
  - `transformers`
  - `datasets`
  - `accelerate`
  - `peft`
  - `trl`
  - `tokenizers`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `tqdm`
  - `pydantic`
  - `pyyaml`
  - `rich`
  - `wandb` または `tensorboard`
  - `pytest`
  - `ruff`
- パッケージ構成を追加する。
  - `src/llm_emotion_test/config.py`
  - `src/llm_emotion_test/data/`
  - `src/llm_emotion_test/models/`
  - `src/llm_emotion_test/training/`
  - `src/llm_emotion_test/agents/`
  - `src/llm_emotion_test/tasks/`
  - `src/llm_emotion_test/evaluation/`
  - `src/llm_emotion_test/cli/`
- 設定ファイル置き場を追加する。
  - `configs/base.yaml`
  - `configs/sft.yaml`
  - `configs/distill.yaml`
  - `configs/rl_grpo.yaml`
  - `configs/eval.yaml`
- 出力ディレクトリの規約を決める。
  - `outputs/runs/<run_id>/config.yaml`
  - `outputs/runs/<run_id>/metrics.jsonl`
  - `outputs/runs/<run_id>/checkpoints/`
  - `outputs/runs/<run_id>/samples.jsonl`
- CLI の入口を整備する。
  - `llm-emotion-test prepare-data`
  - `llm-emotion-test train-sft`
  - `llm-emotion-test distill`
  - `llm-emotion-test train-rl`
  - `llm-emotion-test evaluate`
  - `llm-emotion-test sample-dialogue`

### 完了条件

- `uv run llm-emotion-test --help` が動作する。
- 設定ファイルを読み込み、バリデーションエラーを分かりやすく表示できる。
- 最小単位の pytest が通る。

## フェーズ 1: データセット準備

### 目的

GoEmotions などの感情ラベル付きデータセットを、SFT と蒸留に使える形式へ変換する。

### 作業内容

- Hugging Face Datasets から GoEmotions を読み込む処理を実装する。
- 感情ラベルの正規化を実装する。GoEmotionsは感情ラベルの種類が多いので，8種類くらいに集約する．
  - GoEmotions の複数ラベルを保持する。
  - 必要に応じて代表ラベルへ縮約できる設定を用意する。
  - neutral や複数感情サンプルの扱いを設定可能にする。
- 学習サンプル形式を定義する。
  - `input_text`
  - `target_text`
  - `emotion_labels`
  - `input_latent_id`
  - `target_latent_id`
  - `split`
- soft prompt ID の割り当て規則を実装する。
  - 感情ラベルごとに固定 ID を割り当てる方式。
  - 現在の入力 soft prompt をそのまま出力する方式。
  - 全 soft prompt からサンプリングする方式。
- データ変換結果を JSONL または Arrow で保存する。
- データ品質確認用の集計 CLI を追加する。
  - ラベル分布
  - テキスト長分布
  - latent ID 分布
  - split ごとの件数

### 完了条件

- `uv run llm-emotion-test prepare-data --config configs/sft.yaml` でデータが生成される。
- 小さいサンプル数に制限して smoke test 用データを作れる。
- 変換ロジックに対する単体テストがある。

## フェーズ 2: soft prompt 対応モデル層

### 目的

Qwen 系 causal LM に learnable soft prompt を注入し、出力末尾の再帰 soft prompt 指定を扱えるようにする。

### 作業内容

- Hugging Face `AutoModelForCausalLM` と `AutoTokenizer` のロード処理を実装する。
- soft prompt モジュールを実装する。
  - `num_latents`
  - `prompt_length`
  - `hidden_size`
  - `init_strategy`
- 入力系列へ soft prompt embedding を挿入する forward wrapper を実装する。
- LoRA / QLoRA を設定で有効化できるようにする。
- 出力末尾の latent 指定用特殊トークンを設計する。
  - 例: `<|emotion_latent:003|>`
  - tokenizer へ特殊トークンとして追加する。
  - 生成結果から latent ID を抽出する parser を実装する。
- latent ID が不正な場合の fallback を実装する。
  - 前回 latent を維持する。
  - neutral latent に戻す。
  - 設定に応じてエラーにする。
- 保存・復元処理を実装する。
  - base model ID
  - tokenizer
  - LoRA adapter
  - soft prompt weights
  - emotion label mapping

### 完了条件

- ダミー入力に対して soft prompt 付き forward が通る。
- 特殊トークン付き生成結果から latent ID を抽出できる。
- checkpoint を保存し、再ロード後に同じ形状で推論できる。

## フェーズ 3: 感情 SFT

### 目的

感情ラベル付きデータで、入力 soft prompt と出力 latent marker を含む SFT を行う。

### 作業内容

- SFT 用 dataset class / data collator を実装する。
- 入力テンプレートを定義する。
  - ユーザー入力
  - 現在の latent ID
  - 応答本文
  - 次回 latent marker
- loss 計算対象を制御する。
  - 入力 prompt 部分は loss から除外する。
  - 応答本文と latent marker を loss 対象にする。
- training loop を実装する。
  - まずは `transformers.Trainer` または `trl.SFTTrainer` を使う。
  - gradient accumulation, fp16/bf16, checkpointing を設定化する。
- 評価指標を実装する。
  - validation loss
  - latent marker accuracy
  - 感情分類器による生成文の感情一致率
  - サンプル生成ログ
- 小規模 smoke test を実装する。
  - tiny model または小さい Qwen モデル
  - 10-100 サンプル
  - 1-2 step の学習

### 完了条件

- `uv run llm-emotion-test train-sft --config configs/sft.yaml` が checkpoint を出力する。
- SFT 後モデルで、入力 latent に応じて応答と次 latent marker を生成できる。
- smoke test が CI 的に短時間で通る。

## フェーズ 4: 教師モデルによる自己蒸留

### 目的

教師モデルに自然言語の感情指示を与え、学生モデルが soft prompt で同等の応答分布と latent 遷移を再現できるようにする。

### 作業内容

- 教師プロンプトテンプレートを実装する。
  - 例: 「怒りっぽく返答してください」
  - 例: 「悲しげに、しかし協力的に返答してください」
- 教師モデル生成パイプラインを実装する。
  - temperature
  - top_p
  - max_new_tokens
  - batch generation
  - キャッシュ保存
- 蒸留データ形式を定義する。
  - `base_input_text`
  - `teacher_instruction`
  - `teacher_output_text`
  - `emotion_label`
  - `student_input_latent_id`
  - `student_target_latent_id`
- 学生モデル学習を実装する。
  - 教師出力テキストへの SFT loss
  - latent marker prediction loss
  - 任意で KL divergence loss を追加できる設計にする。
- 教師生成の品質フィルタを実装する。
  - 空応答除外
  - 過度に長い応答の truncate
  - latent marker 形式の検証
  - 重複除去
- 蒸留済みモデルの比較評価を実装する。
  - SFT のみモデル
  - 蒸留モデル
  - 教師モデル

### 完了条件

- `uv run llm-emotion-test distill --config configs/distill.yaml` で蒸留データ生成と学生モデル学習を実行できる。
- 教師生成は再実行時にキャッシュを再利用できる。
- 同一入力に対して感情 latent を切り替えた比較サンプルを出力できる。

## フェーズ 5: 複数エージェント交渉環境

### 目的

複数 LLM がテキストと latent をやり取りし、交渉を通じてスコア最大化を目指す環境を作る。

### 作業内容

- 交渉タスクを選定して実装する。
  - 最初の候補は item division / bargaining task とする。
  - 各エージェントは非公開の価値関数を持つ。
  - 対話後に合意案を提出し、スコアを計算する。
- environment API を定義する。
  - `reset(seed)`
  - `step(agent_action)`
  - `observe(agent_id)`
  - `compute_reward()`
  - `is_done`
- action 形式を定義する。
  - `message_text`
  - `next_latent_id`
  - `proposal`
- 対話プロトコルを実装する。
  - turn-based
  - 最大ターン数
  - 合意/拒否/終了の特殊行動
- transcript 保存形式を実装する。
  - 各ターンの発話
  - latent ID
  - proposal
  - reward
  - task state
- ルールベース agent を実装する。
  - ランダム提案 agent
  - greedy agent
  - fixed emotion latent agent
- LLM agent wrapper を実装する。
  - observation から prompt を作る。
  - 生成テキストから message, proposal, latent を parse する。

### 完了条件

- LLM なしのルールベース agent 同士で episode を実行できる。
- 学習済み SFT/蒸留モデルを agent として接続できる。
- transcript と reward が JSONL に保存される。

## フェーズ 6: GRPO による強化学習

### 目的

交渉タスクの報酬に基づき、タスク達成に有効な感情表現と latent 遷移を学習する。

### 作業内容

- GRPO 学習設計を具体化する。
  - 1 つの task instance に対して複数 rollout を生成する。
  - rollout group 内で相対 advantage を計算する。
  - policy model と reference model の KL penalty を入れる。
- `trl` の GRPO 実装が利用可能であれば優先して統合する。
- 既存実装で multi-agent transcript を扱いづらい場合は、最小 GRPO trainer を自前実装する。
- reward 関数を実装する。
  - 合意成立報酬
  - 個別スコア
  - 全体効用
  - 不正 proposal penalty
  - 長すぎる対話 penalty
- 学習対象を設定化する。
  - LoRA adapter
  - soft prompt weights
  - latent transition head
  - 必要に応じて一部 transformer layer
- rollout buffer を実装する。
  - prompt
  - generated tokens
  - logprobs
  - latent IDs
  - rewards
  - group IDs
- 定期評価を実装する。
  - fixed seed tasks
  - fixed baseline agents
  - reward trend
  - agreement rate
  - latent usage entropy
- checkpoint と resume を実装する。

### 完了条件

- `uv run llm-emotion-test train-rl --config configs/rl_grpo.yaml` で短い GRPO smoke run ができる。
- RL 前後で交渉成功率、平均 reward、latent 使用分布を比較できる。
- 不正な生成や parser failure が学習全体を停止させず、ログに記録される。

## フェーズ 7: 評価・観察・可視化

### 目的

学習された感情表現が、単なる外部制御ではなく、対話履歴やタスク状態に応じて内的に遷移しているかを観察する。

### 作業内容

- 評価 dataset / task seeds を固定する。
- 比較対象を用意する。
  - base model
  - SFT model
  - distilled model
  - RL model
  - latent fixed ablation
  - random latent ablation
  - no latent ablation
- 定量評価を実装する。
  - 平均 reward
  - 合意率
  - Pareto efficiency
  - fairness
  - 発話長
  - parser failure rate
  - latent transition entropy
  - task state と latent の相互情報量
  - emotion classifier による表出感情分布
- 定性評価を実装する。
  - transcript sampling
  - 同一 task でのモデル比較
  - latent trajectory の表示
  - 交渉失敗例の抽出
- 可視化スクリプトを追加する。
  - metrics CSV 出力
  - latent transition heatmap
  - reward curve
  - emotion distribution plot
- 実験レポート生成用の Markdown 出力を実装する。

### 完了条件

- `uv run llm-emotion-test evaluate --config configs/eval.yaml` で評価結果が `outputs/` に出力される。
- RL 前後の違いを、数値と transcript の両方で確認できる。
- ablation により latent の寄与を比較できる。

## フェーズ 8: 品質保証と再現性

### 目的

研究実験として再実行可能で、失敗時に原因を追いやすい状態にする。

### 作業内容

- random seed を一元管理する。
- すべての run で設定ファイル、git commit、依存関係、モデル ID を保存する。
- 主要処理にテストを追加する。
  - config validation
  - dataset conversion
  - latent marker parser
  - soft prompt shape
  - negotiation environment
  - reward function
  - transcript serialization
- lint / format 設定を追加する。
- README.md に実行手順を追記する。
  - セットアップ
  - smoke test
  - SFT
  - 蒸留
  - RL
  - 評価
- 大きな成果物が git に入らないよう `.gitignore` を確認する。
- GPU メモリ不足時の回避策を設定例として追加する。

### 完了条件

- `uv run pytest` が通る。
- `uv run ruff check .` が通る。
- smoke test の一連の流れを README.md の手順通りに再現できる。

## 推奨実装順序

1. CLI と設定読み込みを作る。
2. データ変換と latent marker parser を作る。
3. soft prompt wrapper を単体で動かす。
4. 小規模 SFT を動かす。
5. 教師生成キャッシュと蒸留を動かす。
6. ルールベース交渉環境を動かす。
7. LLM agent を交渉環境へ接続する。
8. 短い GRPO smoke run を動かす。
9. 評価と ablation を整える。
10. README.md とテストを仕上げる。

## 初期の実装スコープ

最初の Codex 実装では、全機能を一度に完成させず、以下を最小到達点にする。

- `prepare-data` で GoEmotions の小規模サンプルを変換できる。
- latent marker parser のテストがある。
- Qwen 互換の tokenizer / model loader がある。
- soft prompt embedding を挿入する wrapper の shape test がある。
- `train-sft` の smoke run が 1-2 step だけ動く。

この段階まで到達した後、蒸留、交渉環境、GRPO の順に実装を広げる。

## リスクと対策

- Qwen3.5 というモデル名が Hugging Face 上で安定していない可能性がある。
  - 設定では `model_name_or_path` を外部指定にし、初期値は利用可能な Qwen 系 instruct/base model にする。
- soft prompt と Hugging Face Trainer の統合が複雑になる可能性がある。
  - まず wrapper の forward を単体テストし、その後 Trainer 用 collator と統合する。
- multi-agent GRPO は既存 trainer と相性が悪い可能性がある。
  - 最初は single policy が両 agent を共有する self-play として実装し、必要なら後で agent ごとに policy を分ける。
- 生成結果の proposal parser が不安定になる可能性がある。
  - JSON 形式の proposal を要求し、失敗時 penalty と fallback を明示する。
- 感情 latent が単なるラベル暗記になる可能性がある。
  - fixed/random/no latent ablation と task state-latent 相互情報量を必ず評価する。
- 学習コストが大きくなる可能性がある。
  - smoke config、LoRA/QLoRA、データ件数制限、短い rollout を標準で用意する。

## 成果物

- 実行可能な CLI パイプライン。
- 感情 soft prompt 対応モデル wrapper。
- SFT / 蒸留 / GRPO の学習スクリプト。
- 交渉タスク環境と LLM agent。
- 評価・可視化スクリプト。
- smoke test と単体テスト。
- 再現手順を含む README.md。
