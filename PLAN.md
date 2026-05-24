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
- soft promptは出力テキスト末尾に専用タグで挟み込んで出力できるようにする．これは，文字にデコードするのではなく，そのままLatentとして再帰的に入力する．

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
  - `wandb`
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

感情ラベル付きデータセット WRIME を、SFT と蒸留に使える形式へ変換する。
soft promptのtoken数は設定で変更できるようにする（デフォルト4）

### 作業内容

- WRIME を読み込む処理を実装する。
- 感情ラベルの正規化を実装する。
  - 必要に応じて代表ラベルへ縮約できる設定を用意する。いったんオリジナルの8種類で良い．
- 学習サンプル形式を定義する。
  - `input_text`
  - `target_text`
  - `emotion_labels`
  - `input_latent_id`
  - `target_latent_id`
  - `split`
- 感情ラベルごとに入力 soft prompt を切り替えるようにする．つまり，感情ラベル数だけsoft promtが存在する．このsoft promptは<|emotion|>タグで挟む．
- また，出力についてもテキスト出力時に <|emotion|>タグで挟んで，感情soft promptを出力するようにする．この時，出力は以下の2パターンから確率的に選択する．
  - 全 soft prompt から一様サンプリングする方式。
  - 入力 soft promptをそのまま出力する方式
- データ変換結果を JSONL で保存する。
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

Qwen3.5 2B (Hugging Face: Qwen/Qwen3.5-2B) に learnable soft prompt を注入し、出力末尾の再帰 soft prompt 指定を扱えるようにする。

### 作業内容

- Hugging Face `AutoModelForCausalLM` と `AutoTokenizer` のロード処理を実装する。
- soft prompt モジュールを実装する。
  - `num_latents`
  - `prompt_length`
  - `hidden_size`
  - `init_strategy`
- 入力系列へ soft prompt embedding を挿入する forward wrapper を実装する。
- LoRA / QLoRA を設定で有効化できるようにする。
- 出力末尾の latent 指定用特殊トークンを設計する。 <|emotion|> など．
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
  - 10-100 サンプル
  - 1-2 step の学習

### 完了条件

- `uv run llm-emotion-test train-sft --config configs/sft.yaml` が checkpoint を出力する。
- SFT 後モデルで、入力 latent に応じて応答と次 latent marker を生成できる。
- smoke test が CI 的に短時間で通る。

## フェーズ 4: 教師モデルによる自己蒸留

### 目的

教師モデルに自然言語の感情指示を与え、学生モデルが soft prompt で同等の応答分布と latent 遷移を再現できるようにする。
教師モデルには Qwen/Qwen3.5-9B のような大きめのモデルを選択できるようにする．
教師モデルには、テキスト入力に加えて、感情表現の指示（怒りっぽく出力して）などを加える。
学生モデルには、テキスト入力に加えて、learnableなsoft promptを入力し、教師モデルの出力latentを教師信号として利用する。その時に、教師側に与えた感情ラベルに応じてsoft promptを切り替える。また、学生モデルはテキスト出力末尾に、必ず再帰入力用のsoft promptを出力するように学習する。
整理すると以下のようになる．
- 教師モデルへの入力：<テキスト入力><感情表現指示><日本語出力指示>
- 学生モデルへの入力：<テキスト入力><感情soft prompt>
- 教師モデルの出力：<テキスト出力latent>
- 学生モデルの出力：<テキスト出力latent><感情soft prompt>

### 作業内容

- 教師プロンプトテンプレートを実装する。
  - 例: 「怒りっぽく返答してください」
  - 例: 「悲しげに、しかし協力的に返答してください」
  - この時，教師モデルは日本語を出力するようにする．
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

複数 LLM がテキストをやり取りし、交渉を通じてスコア最大化を目指す環境を作る。
各対話ステップにおいて暫定的な回答を出力し，評価可能にする．

### 作業内容

- Cooperative Hidden Constraintsタスクを実装する。

まず、**隠れ制約付きの協調コード推定タスク**を作ります。各問題には正解となる4桁または5桁のコードを1つ設定します。各桁は0〜9の数字です。

次に、その正解コードを一意に特定できる制約集合を生成します。制約は、桁の偶奇、桁同士の大小関係、合計値、特定桁の禁止値、全桁が異なること、ある桁が別の桁より1大きいこと、などです。

その制約集合を2つに分割します。Agent Aには一部の制約だけを渡し、Agent Bには残りの制約だけを渡します。A単独、B単独では正解を一意に特定できず、AとBの情報を合わせると一意に特定できるようにします。

各エージェントには、自分のprivate constraintsだけを含むプロンプトを与えます。相手の制約は見えません。エージェント同士は数ターンだけ会話できます。会話の目的は、互いの制約を共有し、正解コードを共同で推定することです。

各会話ターン終了時（つまり，エージェントAとエージェントBが1回ずつテキスト出力を完了した時点）で，第三者のLLMがそのターンにおける暫定的な回答を出力します．

- 例
```text
<answer>1234</answer>
```
ここら辺の出力形式はQwenのreasoning形式に合わせる．

評価器は、回答を読み取り、次の項目を採点します。
- 正解コードと一致したか。
- エージェントA・Bの回答形式が制約をどれだけ満たしているか。つまり，感情latentを<|emotion|>タグで挟み込んで出力できているか，など．問題生成の制約条件の事ではない．

GRPOでは、同じ問題に対して複数の会話ロールアウトを生成します。それぞれのロールアウトに報酬を付け、同一問題内で相対比較します。高報酬の会話・推論・最終回答が強化されます。
GRPOで学習を行うのはエージェントA・Bのみであり，第三者のLLMは学習されません．

問題生成器の要件
- 入力パラメータ

| 項目         | 内容                   |
| ---------- | -------------------- |
| 桁数         | 例: 4桁、5桁             |
| 使用可能な数字    | 通常は0〜9               |
| 重複許可       | 桁の重複を許すか             |
| 難度         | easy / medium / hard |
| Agent数     | MVPでは2               |
| 各Agentの制約数 | 最小・最大                |
| 候補数条件      | 各Agent単独で何候補以上残すか    |
| 情報量バランス条件  | A/B単独の候補数差をどこまで許すか   |
| 使用する制約タイプ  | 偶奇、大小、合計、差分、集合、個数など  |

- 出力

1問ごとに以下を出す。

| 項目         | 内容                   |
| ---------- | -------------------- |
| 正解コード      | 一意解                  |
| Agent Aの制約 | Aだけが見る制約リスト          |
| Agent Bの制約 | Bだけが見る制約リスト          |
| 全制約        | 採点用                  |
| メタ情報       | A単独候補数、B単独候補数、全制約候補数 |

- 制約タイプ

MVPで実装する制約。

| 種類       | 例                |
| -------- | ---------------- |
| 桁の偶奇     | 2桁目は偶数           |
| 桁の大小     | 1桁目 < 3桁目        |
| 桁同士の差    | 4桁目 = 2桁目 + 1    |
| 全桁の合計    | 各桁の合計は17         |
| 特定桁の禁止値  | 1桁目は0, 5, 9ではない  |
| 特定桁の候補集合 | 3桁目は2, 4, 7のいずれか |
| 全桁の重複有無  | 全桁は異なる           |
| 値の出現有無   | 7を含む / 0を含まない    |
| 偶数・奇数の個数 | 偶数は2個            |

- 生成アルゴリズム

1. 正解コードをランダム生成する。
2. 正解コードを満たす制約候補を列挙する。
3. 制約候補から一部を選ぶ。
4. 全候補コードに対して制約集合を評価する。
5. 候補が一意解になるまで制約を追加する。
6. 一意解にならない場合は作り直す。
7. 制約集合をAgent A/Bに分割する。
8. A単独、B単独では一意解にならないことを確認する。
9. A単独候補数とB単独候補数が条件範囲内か確認する。
10. 制約タイプの偏りを確認する。
11. 条件を満たしたものを採用する。

- 採用条件

生成した問題は以下をすべて満たす必要がある。

| 条件     | 内容                    |
| ------ | --------------------- |
| 正解性    | 正解コードが全制約を満たす         |
| 一意性    | A+Bの全制約で解が1つに定まる      |
| 情報非対称性 | A単独では解けない             |
| 情報非対称性 | B単独では解けない             |
| 協調必要性  | A/Bの情報共有がないと高スコアにならない |
| バランス   | A/Bの候補数が極端に偏らない       |
| 多様性    | 同じ制約タイプだけに偏らない        |
| サイズ制限  | 制約数が多すぎない             |
| 明確性    | 制約文が一意に解釈できる          |

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
- transcript 保存形式を実装する。
  - 各ターンの発話
  - latent ID
  - proposal
  - reward
  - task state
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
各モデルは自身が出力したemotion latentをsoft promptとして再帰的に入力する．
モデルAとモデルBはemotion latentを交換することはできず，出力テキストだけを対話形式で交換する．
最適な感情表現を獲得する事を目指す．

### 作業内容

- docs/at-grpo.pdf を参照して，AT-GRPOアルゴリズムを実装し，ターンごとの評価を可能にする．
- reward 関数を実装する。
- 学習対象を設定する。
  - LoRA adapter
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