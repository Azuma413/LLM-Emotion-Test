# Emotion latent を文字列 CE ではなく latent loss で学習する修正計画

## 背景

現在の SFT / Distill 実装では、`target_text` の末尾にある `<|emotion|>001<|/emotion|>` のような emotion marker が通常のテキスト token として tokenization され、その token 列が causal LM の cross entropy loss 対象になっている。

この方式では、モデルは「次の emotion latent」を latent 空間で直接予測しているのではなく、あくまで marker 文字列を生成するように学習している。データセットやログ上は marker 文字列を保持してよいが、学習信号としては `target_latent_id` に対応する soft prompt / latent 表現へ回帰する loss に置き換える。

## 方針

推奨する修正は、以下の hybrid 方式にする。

1. JSONL では引き続き `target_text` に emotion marker を保持する。
2. 学習 collator では `target_text` から末尾の emotion marker を parse して取り除く。
3. response 本文には従来通り token-level cross entropy loss をかける。
4. response 本文末尾、または専用 anchor token の hidden state から `target_latent_id` を予測する。
5. 予測 latent と、`target_latent_id` に対応する soft prompt embedding を比較する regression loss を追加する。
6. 最終 loss は SFT では `text_ce_loss + latent_loss_weight * latent_regression_loss` とし、Distill ではさらに teacher logits への KL loss を足す。

この設計にすると、データ形式と既存の出力ログは大きく壊さずに、学習信号だけを「marker 文字列」から「latent 表現」へ移せる。

## 重要な設計判断

### marker token は CE 対象から外す

SFT / Distill の collator で、現在は以下のように `target_ids` 全体が labels に入っている。

```python
labels = [-100] * len(prompt_ids) + target_ids
```

修正後は、`target_text` 末尾の marker を除いた `response_text` だけを `target_ids` にする。marker token は input に含めないか、anchor token として入れる場合も labels は `-100` にする。

### latent 予測位置

最初は専用 anchor token を使うのが実装しやすい。

例:

```text
input_ids = prompt_ids + response_ids + [latent_pred_token_id]
labels    = -100...    + response_ids + [-100]
```

`latent_pred_token_id` の位置の hidden state を `latent_head` に渡して、次 latent を予測する。anchor token は生成させたいテキストではなく、latent supervision のための内部スロットとして使う。

anchor token を使いたくない場合は、response 最終 token の hidden state から予測してもよい。ただし空 response や truncation 時の扱いが面倒になるため、まずは anchor token 方式を優先する。

### regression target

`target_latent_id` に対応する soft prompt は shape が `[prompt_length, hidden_size]` なので、以下のどちらかにする。

- `mean`: soft prompt の prompt_length 次元を平均し、`[hidden_size]` へ回帰する。
- `flatten`: `[prompt_length * hidden_size]` へ flatten して全 soft prompt に回帰する。

初期実装は `mean` を推奨する。軽く、latent head も小さく、response hidden state からの予測として安定しやすい。

loss はまず以下の組み合わせにする。

```text
latent_regression_loss = mse(normalize(pred), normalize(target_soft_prompt_mean.detach()))
```

`detach()` を入れる理由は、latent regression loss によって soft prompt 側まで自由に動くと、予測器と target codebook が一緒に動いて collapse しやすいため。soft prompt 自体は入力条件として text CE からは引き続き学習される。

必要なら後段で cosine loss や auxiliary latent classification loss を追加できるようにする。

## 実装ステップ

### 1. config を追加する

`src/llm_emotion_test/config.py` に latent supervision 用設定を追加する。

候補:

```yaml
latent_training:
  mode: regression          # marker_ce | regression | regression_plus_marker_ce
  loss_weight: 1.0
  target: soft_prompt_mean  # soft_prompt_mean | soft_prompt_flatten
  detach_target: true
  normalize: true
  anchor_token: "<|latent_pred|>"
  auxiliary_classification_weight: 0.0
```

互換性のため、既存挙動を残したい場合は default を `marker_ce` にする。今回の研究方針を標準にするなら `regression` を default にする。

### 2. tokenizer / model loader で anchor token を追加する

`src/llm_emotion_test/models/latent.py` または `models/loader.py` で `<|latent_pred|>` を tokenizer に追加する。

既存の `<|emotion|>` / `<|/emotion|>` special token 追加処理とは別に、latent prediction anchor 用 token を追加する。

追加後は `resize_token_embeddings(len(tokenizer))` が呼ばれることを確認する。

### 3. collator を latent-aware にする

`EmotionSFTDataCollator` を修正するか、新しく `LatentRegressionDataCollator` を追加する。

責務:

- `feature["target"]` から末尾 marker を取り除いた `response_text` を作る。
- `target_latent_id` を batch tensor に入れる。
- `input_ids` は `prompt_ids + response_ids + [anchor_token_id]` にする。
- `labels` は `[-100] * len(prompt_ids) + response_ids + [-100]` にする。
- `latent_position` として anchor token の sequence index を返す。

返す batch 例:

```python
{
    "input_ids": ...,
    "attention_mask": ...,
    "labels": ...,
    "latent_ids": input_latent_ids,
    "target_latent_ids": target_latent_ids,
    "latent_positions": anchor_positions,
}
```

既存の `target_latent_id` は dataset item にすでに含まれているので、JSONL schema 自体は変更しなくてよい。

### 4. SoftPromptCausalLM に latent prediction head を追加する

`src/llm_emotion_test/models/soft_prompt.py` の `SoftPromptCausalLM` に以下を追加する。

- `latent_head: nn.Linear(hidden_size, hidden_size)` から始める。
- `forward(..., target_latent_ids=None, latent_positions=None)` を受け取れるようにする。
- base model forward で `output_hidden_states=True` を使い、最終 hidden state を取り出す。
- soft prompt prepend 分だけ position がずれるため、`latent_positions + soft_prompt.prompt_length` の hidden state を gather する。
- `pred_latent = latent_head(anchor_hidden)` を計算する。
- `target_latent = soft_prompt(target_latent_ids).mean(dim=1).detach()` を作る。
- `latent_loss` を MSE/cosine で計算する。
- `outputs.loss = outputs.loss + weight * latent_loss` 相当の返却にする。

Hugging Face の `Trainer` と相性を保つため、返り値は `loss`, `logits`, `latent_loss`, `text_loss`, `pred_latent` を持つ object にするか、既存 output に loss だけ差し替える薄い wrapper を用意する。

### 5. DistillationTrainer の KL loss と整合させる

`src/llm_emotion_test/training/distill.py` の `DistillationTrainer.compute_loss` は現在 `outputs.loss` に任意 KL を足している。自己蒸留段階では KL を標準で有効化し、`configs/distill.yaml` の `distillation.kl_divergence_weight` は `0.1` 以上にする。

修正後:

```text
total_loss = text_ce_loss + latent_loss_weight * latent_loss + kl_weight * kl_loss
```

KL は response token 部分だけにかける。anchor token と marker token には KL をかけない。

既存実装の KL は `kl_divergence_weight > 0.0` のときだけ teacher model をロードする。設定変更後は Distill 実行時に teacher generation 用モデルに加えて KL teacher model も必要になるため、GPU メモリ使用量が増える点を明記する。メモリが厳しい場合は、teacher generation と KL teacher を同一ロードで再利用する設計へ後続改善する。

### 6. generation / evaluation を latent head 対応にする

現在の `generate_sft_samples` は生成テキスト中の marker を parse して `predicted_latent_id` を得ている。

修正後は以下にする。

1. モデルで response text を generate する。
2. 生成された response の末尾に anchor token を付けて再度 forward する、または generate 時の hidden state を使う。
3. latent head の出力と soft prompt codebook の距離を計算し、nearest latent ID を `predicted_latent_id` にする。
4. ログや互換出力では `format_latent_marker(predicted_latent_id)` を response の末尾に付けてもよい。

これにより、外部から見る JSONL や transcript は引き続き `<|emotion|>...` を持てるが、モデル内部の予測は latent head 経由になる。

### 7. RL / negotiation への接続

`LLMNegotiationAgent` と RL rollout は現在、生成テキスト中の marker を parse して `next_latent_id` を決めている。

修正後の方針:

- agent の `act` は `generated_text` と `predicted_next_latent_id` を別々に得る。
- `AgentAction.next_latent_id` には latent head の nearest latent を入れる。
- transcript 用 `raw_text` には、必要なら `generated_text + marker(predicted_next_latent_id)` を入れる。
- `parse_agent_action` は backward compatibility 用に残す。
- RL の token logprob は response text の token に対して計算し、latent head には別途 RL 用 objective を後で追加する。

初期対応では、RL 前の SFT/Distill checkpoint が latent head を持つこと、かつ agent が marker parse ではなく latent head を使えることを完了条件にする。RL の latent head に policy gradient を直接かける設計は次段階でよい。

### 8. tests を更新する

追加・修正するテスト:

- SFT collator が marker を labels に含めないこと。
- collator が `target_latent_ids` と `latent_positions` を返すこと。
- `SoftPromptCausalLM.forward` が `target_latent_ids` 付きで `latent_loss` を返すこと。
- `latent_loss_weight=0` で既存 CE と同等に動くこと。
- Distill student dataset でも同じ collator が使えること。
- sample generation が marker parse ではなく latent head nearest で `predicted_latent_id` を出すこと。
- checkpoint save/load で latent head の重みが保存復元されること。

## 互換性と移行

- JSONL の `target_text` は当面そのまま marker 付きで保存する。
- 学習時だけ marker を parse して loss から外す。
- 既存 checkpoint は latent head を持たないため、ロード時に head を新規初期化できるようにする。
- `sample_latent_marker_accuracy` は名前を `sample_latent_accuracy` に変更する。互換表示として旧 key も metrics に残してよい。
- `USAGE.md` は「marker 文字列を教師 token として学習する」説明から、「marker はデータ表現で、学習時は target_latent_id に対する latent regression loss に変換される」説明へ更新する。

## 完了条件

- `uv run pytest` が通る。
- `prepare-data` の出力 schema は維持される。
- `train-sft` で text CE と latent regression loss が metrics に記録される。
- `distill` で教師生成テキストへの CE と `student_target_latent_id` への latent regression loss が同時に使われる。
- `distill` で teacher logits への KL loss が有効になり、metrics に `kl_divergence_weight` と可能なら `kl_loss` が記録される。
- 生成サンプルに `predicted_latent_id`, `target_latent_id`, `latent_distance` が出力される。
- marker token が labels に含まれていないことをテストで保証する。

## 実装順

1. config と tokenizer anchor token を追加する。
2. marker strip / target latent extraction helper を実装する。
3. latent-aware collator を追加し、SFT/Distill trainer に接続する。
4. `SoftPromptCausalLM` に latent head と loss 計算を追加する。
5. checkpoint save/load を latent head 対応にする。
6. sample generation / metrics を latent head 対応にする。
7. RL agent が marker parse ではなく latent head から next latent を取れるようにする。
8. tests と USAGE.md を更新する。
