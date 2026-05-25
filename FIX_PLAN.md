# 修正プラン

## 背景

現状実装は、soft prompt を latent として入力している一方で、テキスト prompt にも `現在のlatent ID` や latent marker 出力指示を明示している。これは latent を soft prompt として扱う設計と重複しており、モデルが soft prompt を使わずにテキスト上の説明だけで課題を解く余地を作っている。

また、SFT では WRIME の同じ文章を入力と出力の両方に使っているため、学習課題が「感情条件付き応答生成」ではなく「入力文コピー」に近い。これでは soft prompt が感情状態として効いているかを評価しづらく、後段の自己蒸留や GRPO の初期化としても弱い。

この修正では、SFT を以下の causal LM 形式へ変更する。

```text
input:
  soft_prompt(emotion) + 文の前半

target:
  文の後半 + <|emotion|>next_latent<|/emotion|>
```

`next_latent` は自己蒸留と同様に、入力 latent に紐づく emotion label からルールベースで決定し、教師モデルに latent marker を生成させない。

## 目標

- latent ID はテキストとして入力せず、常に soft prompt embedding としてのみ入力する。
- SFT は同一文コピーではなく、WRIME 文の前半から後半を生成する continuation task にする。
- SFT target の末尾にはルールベースで `<|emotion|>xxx<|/emotion|>` を付与する。
- Qwen Instruct に不要な独自区切り文言、特に `ユーザー入力:`、`応答:`、`現在のlatent ID:` を SFT/蒸留/RL から取り除く。
- 教師モデルには latent marker や emotion tag を生成させない。
- 学生モデルは `soft_prompt + text` を入力し、`teacher_output_text + latent marker` を target として学習する。
- GRPO の A/B agent prompt から、A/B が直接回答する指示と latent marker 出力指示を削除する。
- 評価・サンプル生成・テストも新しい形式に合わせて更新する。

## 現状の問題点

### SFT

対象ファイル:

- `src/llm_emotion_test/data/wrime.py`
- `src/llm_emotion_test/training/sft.py`
- `tests/test_wrime_data.py`
- `tests/test_sft_training.py`

問題:

- `input_text` と `target_text` がどちらも元文全体を含む。
- `build_sft_prompt()` が `現在のlatent ID: ...` をテキストに含めている。
- `build_sft_prompt()` が `ユーザー入力:` / `応答:` という独自形式を使っている。
- loss は target 部分だけに掛かるが、target が元文コピーなので学習信号が弱い。
- latent の教師信号は marker token の生成だけであり、本文と next latent の対応が弱い。

### 自己蒸留

対象ファイル:

- `src/llm_emotion_test/data/distill.py`
- `src/llm_emotion_test/training/distill.py`
- `tests/test_distill_data.py`

問題:

- 教師 prompt に `latent marker` を明示している。
- 教師 prompt に「返答本文の最後に、指定されたlatent markerをそのまま1回だけ付けてください。」という指示がある。
- 教師出力に latent marker が含まれることを期待している。
- 学生 prompt に `ユーザー入力:` / `応答:` が含まれている。

### GRPO

対象ファイル:

- `src/llm_emotion_test/agents/negotiation.py`
- `src/llm_emotion_test/training/rl.py`
- `tests/test_negotiation_env.py`
- `tests/test_rl_grpo.py`

問題:

- A/B agent prompt に `<answer>1234</answer>` 形式で回答候補を含める指示がある。
- A/B agent prompt に latent marker 出力指示がある。
- ただし設計上、回答するのは第三者回答器であり、A/B は制約共有だけを行う。
- latent marker 生成能力は SFT/蒸留で獲得済みという前提なので、RL prompt で毎回説明する必要はない。

## 修正方針

### 1. SFT データ形式を continuation task に変更する

WRIME の元文を `prefix` と `continuation` に分割する。

```json
{
  "input_text": "文の前半",
  "target_text": "文の後半\n<|emotion|>001<|/emotion|>",
  "emotion_labels": {"joy": 0, "sadness": 3, "...": 0},
  "input_latent_id": 1,
  "target_latent_id": 1,
  "split": "train",
  "source_text": "元のWRIME全文"
}
```

`source_text` は必須ではないが、デバッグ・品質集計・サンプル確認のために保存する。

分割方法:

- まず文字列長ベースで文を前半/後半に分割する。
- 空白や句読点が近くにある場合は、そこを優先して切る。
- 極端に短い文は以下のどちらかで扱う。
  - smoke test を優先するなら、最小1文字以上の prefix/target に強制分割する。
  - 品質を優先するなら、短すぎる文を drop する。
- 初期実装では smoke test の安定性を優先し、最小1文字以上に分割する。

候補 API:

```python
def split_text_for_continuation(text: str, *, min_prefix_chars: int = 1, min_target_chars: int = 1) -> tuple[str, str]:
    ...
```

`target_latent_id` の決定:

- 自己蒸留と同様に、入力 latent に紐づく emotion label を基準にする。
- 当面は `target_latent_id = input_latent_id` をデフォルトにする。
- latent 遷移を学習したい場合だけ、別設定で遷移ルールを有効化する。

設定案:

```yaml
data:
  sft_task: continuation
  min_prefix_chars: 1
  min_target_chars: 1
  target_latent_strategy: copy_input
```

既存の `copy_input_latent_probability` は、SFT continuation では基本的に使わない。後方互換を残す場合でも、デフォルトは `target_latent_strategy: copy_input` に寄せる。

### 2. SFT prompt から latent ID と独自区切り文言を削除する

現状:

```text
ユーザー入力:
{input_text}

現在のlatent ID: 001
応答:
```

修正後:

```text
{input_text}
```

モデル入力は以下のみになる。

```text
soft_prompt(input_latent_id) + tokenized(input_text)
```

教師信号は以下。

```text
target_text = continuation_text + "\n" + latent_marker(target_latent_id)
```

collator の基本方針は維持する。

- `input_ids = prompt_ids + target_ids`
- `labels = [-100] * len(prompt_ids) + target_ids`
- `latent_ids = input_latent_id`

このため、soft prompt と prefix 部分には loss を掛けず、continuation と latent marker にのみ causal LM loss を掛ける。

### 3. Qwen chat template の扱いを整理する

SFT continuation task は対話ではないため、原則として chat template を使わない。

理由:

- 入力は user utterance ではなく、単一文の prefix である。
- target は assistant 応答ではなく、同一文の continuation である。
- ここで chat template を入れると、continuation task と対話 task が混ざる。

一方、自己蒸留の教師モデルは Instruct model として使うため、教師 prompt 生成では tokenizer の chat template を使う。

方針:

- SFT continuation: raw text prefix を使う。
- Distill teacher: `tokenizer.apply_chat_template()` を使う。
- Distill student: 当面は raw text または teacher prompt と同じ user text を使う。対話応答学習として扱うなら chat template を使う。
- GRPO: 交渉タスク固有 prompt なので chat template を使う余地はあるが、まず不要指示の削除を優先する。

### 4. 自己蒸留の教師 prompt から latent marker 指示を削除する

現状の教師 prompt は marker を出すように指示している。修正後は、教師は自然言語応答のみを生成する。

修正後の教師 messages 例:

```python
[
    {
        "role": "system",
        "content": "あなたは日本語で自然に返答するアシスタントです。"
    },
    {
        "role": "user",
        "content": f"{emotion_instruction}\n\n入力:\n{base_input_text}"
    },
]
```

教師出力:

```text
自然言語応答のみ
```

学生用 target:

```text
{teacher_output_text}
<|emotion|>{student_target_latent_id:03d}<|/emotion|>
```

この marker 付与はルールベースで行う。

変更点:

- `build_teacher_prompt()` から `latent_marker` 引数を削除する。
- `normalize_teacher_record()` で教師出力から latent marker を parse する処理をやめる。
- 教師出力に latent marker が混入した場合は、必要に応じて除去する。
- `require_teacher_latent_marker` は廃止または無視する。
- `DistillationRecord.teacher_output_text` は marker 無しの自然言語応答にする。
- `DistillationRecord.as_student_sft_record()` で marker を付与した `target_text` を作る。

候補 helper:

```python
def append_latent_marker(text: str, *, latent_id: int, marker_template: str) -> str:
    return f"{text.strip()}\n{format_latent_marker(marker_template, latent_id)}"
```

### 5. 自己蒸留の学生 prompt を単純化する

現状:

```text
ユーザー入力:
{input_text}

応答:
```

修正後:

```text
{input_text}
```

または、対話応答として扱う場合:

```text
<chat_template user={input_text} assistant_prefix=True>
```

まずは既存 SFT collator と整合するように raw text を使う。

入力:

```text
soft_prompt(student_input_latent_id) + tokenized(base_input_text)
```

教師信号:

```text
teacher_output_text + "\n" + latent_marker(student_target_latent_id)
```

### 6. GRPO の A/B agent prompt を修正する

現状の削除対象:

```text
回答候補がある場合は <answer>1234</answer> の形式で含めてください。
最後に必ず <|emotion|>000<|/emotion|> の形式で latent を出力してください。
```

修正後の prompt 例:

```text
あなたは協調コード推定タスクのエージェントです。
あなたのID: Agent A
自分の制約:
- ...
会話履歴:
Agent A: ...
Agent B: ...
次の発話では、相手に有用な制約を共有してください。
```

注意:

- A/B は直接回答しない。
- 第三者回答器が会話履歴から回答を推定する。
- モデル出力の末尾 latent marker は、SFT/蒸留で獲得済みの形式として parse する。
- marker が出ない場合の fallback は既存の `invalid_latent_fallback` に従う。

### 7. README / USAGE の説明を更新する

対象ファイル:

- `README.md`
- `USAGE.md`
- 必要なら `PLAN.md`

更新内容:

- SFT は WRIME 文の continuation task であることを明記する。
- SFT input は `soft_prompt + 文の前半`、target は `文の後半 + latent marker` と説明する。
- latent ID はテキスト prompt に含めないと明記する。
- 自己蒸留では教師モデルは marker を出力しないと説明する。
- 学生 target に marker をルールベースで付与すると説明する。
- GRPO では A/B agent は回答せず、制約共有のみ行うと説明する。

## 実装手順

### Step 1: SFT データ生成の変更

- `PreparedSample` に `source_text` を追加する。
- `convert_wrime_row()` で元文を prefix/continuation に分割する。
- `input_text = prefix`
- `target_text = continuation + "\n" + target_marker`
- `target_latent_id = input_latent_id` をデフォルトにする。
- summary に `source_text` は含めなくてよいが、JSONL には保存する。

テスト:

- `tests/test_wrime_data.py`
  - `input_text` が全文ではなく prefix になること。
  - `target_text` が continuation + marker になること。
  - `target_latent_id == input_latent_id` になること。
  - `source_text` に元文が残ること。

### Step 2: SFT prompt の変更

- `build_sft_prompt(record)` を `return str(record["input_text"])` に変更する。
- `現在のlatent ID`、`ユーザー入力:`、`応答:` を削除する。
- `generate_sft_samples()` も同じ prompt を使うため、サンプル出力が continuation 形式になることを確認する。

テスト:

- `tests/test_sft_training.py`
  - prompt に `現在のlatent ID` が含まれないこと。
  - prompt に `ユーザー入力` / `応答` が含まれないこと。
  - labels は prompt 部分が `-100`、target 部分が loss 対象であること。

### Step 3: 自己蒸留データ生成の変更

- `build_teacher_prompt()` を marker 非依存にする。
- 可能なら teacher tokenizer の `apply_chat_template()` を使う。
- `build_teacher_request()` から `latent_marker` を教師 prompt に渡す処理を削除する。
- `normalize_teacher_record()` は marker parse を行わない。
- 教師出力から既存 marker があれば除去する helper を入れる。
- `as_student_sft_record()` で `teacher_output_text + marker` を target にする。

テスト:

- `tests/test_distill_data.py`
  - 教師 prompt に `<|emotion|>` が含まれないこと。
  - 教師 prompt に latent marker 指示文が含まれないこと。
  - student record の `target_text` 末尾に marker が付くこと。
  - teacher cache の `teacher_output_text` は marker 無しで保持されること。

### Step 4: 自己蒸留学生 prompt の変更

- `build_distill_student_prompt(record)` を `return str(record["input_text"])` に変更する。
- SFT collator はそのまま使う。

テスト:

- student prompt に `ユーザー入力` / `応答` が含まれないこと。
- distill training dataset が `latent_id` を soft prompt 用に返すこと。

### Step 5: GRPO prompt の変更

- `build_agent_prompt()` から `<answer>` 指示を削除する。
- `build_agent_prompt()` から latent marker 出力指示を削除する。
- 「相手に有用な制約を共有してください」程度の指示にする。

テスト:

- `tests/test_negotiation_env.py` または新規テストで prompt に `<answer>` が含まれないこと。
- prompt に `<|emotion|>` が含まれないこと。
- 既存 RL smoke tests が通ること。

### Step 6: 設定ファイル更新

対象:

- `configs/sft.yaml`
- `configs/distill.yaml`
- `configs/rl_grpo.yaml`
- `configs/eval.yaml`
- `configs/base.yaml`

追加/変更候補:

```yaml
data:
  sft_task: continuation
  min_prefix_chars: 1
  min_target_chars: 1
  target_latent_strategy: copy_input
```

削除または非推奨:

```yaml
data:
  copy_input_latent_probability: 0.5

distillation:
  require_teacher_latent_marker: false
```

後方互換を考えるなら、すぐに削除せず config parser 側で ignored/deprecated として扱う。

### Step 7: ドキュメント更新

- `USAGE.md` の SFT 説明を continuation task に変更する。
- 自己蒸留の説明から「教師モデルが latent marker を出す」を削除する。
- GRPO の説明に「A/B は回答せず制約共有、第三者回答器が回答」を明記する。

### Step 8: 回帰テスト

最低限:

```bash
uv run pytest tests/test_wrime_data.py tests/test_sft_training.py tests/test_distill_data.py tests/test_negotiation_env.py tests/test_rl_grpo.py
```

余裕があれば:

```bash
uv run pytest
```

## 期待される学習の意味

### SFT の latent 学習

SFT では `input_latent_id` に対応する soft prompt embedding が token embedding の前に挿入される。loss は prefix ではなく、continuation と latent marker にのみ掛かる。

そのため、各 latent embedding は以下の生成を助ける方向に更新される。

- 与えられた prefix から、その感情に対応する文体・語彙の continuation を出す。
- continuation の末尾に、ルールベースで指定された next latent marker を出す。

分類 loss は使わない。latent は「どの出力分布を誘導するか」を causal LM loss から学習する。

### next latent の扱い

初期方針では `next_latent = input_latent` とする。

理由:

- SFT ではまず latent の安定した再帰入力を学ばせる。
- ランダム遷移を混ぜると、本文の感情と marker の対応が崩れやすい。
- 複雑な latent 遷移は自己蒸留または RL で扱う方がよい。

将来的に遷移を学ばせる場合は、以下のような strategy を追加する。

- `copy_input`: 常に入力 latent を出力する。
- `sample_uniform`: 全 latent から一様サンプルする。
- `emotion_transition_table`: 感情遷移表に従ってサンプルする。
- `teacher_emotion`: 教師応答の推定感情から next latent を決める。

## 完了条件

- SFT JSONL が `prefix -> continuation + marker` 形式になっている。
- SFT prompt に latent ID のテキスト表現が含まれない。
- SFT prompt に `ユーザー入力:` / `応答:` が含まれない。
- 教師モデル prompt に latent marker 指示が含まれない。
- 教師 cache の出力は自然言語応答のみである。
- 学生 target にはルールベースで latent marker が付与される。
- GRPO agent prompt に `<answer>` 指示と latent marker 指示が含まれない。
- 既存テストを新仕様に更新し、pytest が通る。
