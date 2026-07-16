# Critique-Conditioned OPD：W&B 指标实现规范

本文档用于指导 Codex 在当前 `critique_opd` 代码中补充 W&B 指标。所有训练曲线默认以 `global_step` 为横坐标，每个 optimizer step 记录一次；评测指标只在执行评测的 step 记录。

---

## 1. 统一定义

对第 \(i\) 条失败轨迹，在第 \(t\) 个 student 生成 token 位置：

- \(\pi_S^{i,t}\)：student 的概率分布。
- \(\pi_{T_0}^{i,t}\)：不包含 critique 的普通 teacher 分布。
- \(\pi_{T_c}^{i,t}\)：包含 critique 特权信息的 teacher 分布。
- \(y_{i,t}\)：student 实际生成的 token。
- \(e_i\)：teacher critique 解析出的首个错误 action，下标从 0 开始。

### 1.1 两个统计范围

所有分布比较指标都分别统计下面两个范围。

#### `prefix_to_error`

从第一个 action 开始，到首个错误 action 结束为止：

```text
Action 1
环境 prompt
Action 2
环境 prompt
...
Error Action
```

只统计 `response_mask == 1` 的 student 生成 token。环境 observation prompt 仅作为上下文，不能参与指标平均。

#### `error_action`

只统计首个错误 action 中的 student 生成 token。

代码需要保存：

```python
response_step_start_indices
response_step_end_indices
```

第 \(e_i\) 个错误 action 的范围为：

```python
error_start = response_step_start_indices[e_i]
error_end = response_step_end_indices[e_i]
```

### 1.2 聚合方式

分布类指标统一使用“先逐轨迹平均，再对轨迹平均”：

```python
trajectory_metric = token_metric[trajectory_mask].mean()
batch_metric = torch.stack(valid_trajectory_metrics).mean()
```

即：

\[
M_{	ext{batch}}
=
rac{1}{B}
\sum_{i=1}^{B}
\left(
rac{1}{N_i}
\sum_{t=1}^{N_i}
M_{i,t}

ight)
\]

不要把整个 batch 的所有 token 直接混在一起平均，否则长轨迹权重更大。

无有效 token 的轨迹不参与该指标平均；同时记录有效轨迹数量。

---

## 2. JS divergence

目标：比较 student 与普通 teacher、critique teacher 的整体分布差异。

### 2.1 分别构造共同 support

对普通 teacher 和 critique teacher 分别与 student 构造 top-k 并集，不使用三方共同并集。

#### Student 与普通 teacher

\[
\mathcal U^{base}_t
=
\operatorname{TopK}(\pi_S)
\cup
\operatorname{TopK}(\pi_{T_0})
\]

#### Student 与 critique teacher

\[
\mathcal U^{critique}_t
=
\operatorname{TopK}(\pi_S)
\cup
\operatorname{TopK}(\pi_{T_c})
\]



对于每一组比较，两个模型都必须取得该并集中所有 token 的真实概率。不能把没有出现在某个模型自身 top-k 中的 token 概率直接设为 0。

然后分别在各自并集内重新归一化：

\[
\hat p_M(v)
=
\frac{p_M(v)}
{\sum_{u\in\mathcal U_t}p_M(u)}
\]

其中 \(M\) 是当前比较中的 student 或 teacher。

- 计算 `JS(student, base_teacher)` 时，在 \(\mathcal U^{base}_t\) 内分别归一化 student 和普通 teacher。
- 计算 `JS(student, critique_teacher)` 时，在 \(\mathcal U^{critique}_t\) 内分别归一化 student 和 critique teacher。
- 不使用 tail bucket。
- 不把三个模型放进同一个三方并集。

如果当前 vLLM 接口只能返回各自 top-k，Codex 需要增加一个能够对并集中指定 token ID gather log-prob 的 scoring 路径，或者在这些选定位置取得完整 logits 后再 gather。

### 2.2 JS 公式

对于两个分布 \(P,Q\)：

\[
M=rac{P+Q}{2}
\]

\[
JS(P,Q)
=
rac{1}{2}KL(P\Vert M)
+
rac{1}{2}KL(Q\Vert M)
\]

每个位置计算：

```text
JS(student, base_teacher)
JS(student, critique_teacher)
```

再按照第 1.2 节逐轨迹、逐 batch 聚合。

### 2.3 W&B 指标

#### 截断轨迹

```text
distribution/js_student_base_prefix_to_error
distribution/js_student_critique_prefix_to_error
distribution/js_critique_effect_prefix_to_error
```

其中：

\[
	ext{js_critique_effect}
=
JS(\pi_S,\pi_{T_c})
-
JS(\pi_S,\pi_{T_0})
\]

大于 0 表示加入 critique 后，teacher 分布平均离 student 更远。

#### 仅错误 action

```text
distribution/js_student_base_error_action
distribution/js_student_critique_error_action
distribution/js_critique_effect_error_action
```
---

## 3. Top-k overlap ratio

每个 student token 位置分别计算 student 与两个 teacher 的 top-k 集合重合率。

### 3.1 公式

\[
	ext{overlap}(S,T)
=
rac{
|\operatorname{TopK}(S)\cap\operatorname{TopK}(T)|
}{k}
\]

范围为 \([0,1]\)。

- 越大：两个模型高概率候选集合越相似。
- 越小：两个模型的主要候选 token 差异越大。

### 3.2 W&B 指标

#### 截断轨迹

```text
distribution/topk_overlap_student_base_prefix_to_error
distribution/topk_overlap_student_critique_prefix_to_error
distribution/topk_overlap_critique_effect_prefix_to_error
```

定义：

\[
	ext{overlap_critique_effect}
=
	ext{overlap}(S,T_c)
-
	ext{overlap}(S,T_0)
\]

小于 0 表示 critique teacher 的 top-k 与 student 重合更少。

#### 仅错误 action

```text
distribution/topk_overlap_student_base_error_action
distribution/topk_overlap_student_critique_error_action
distribution/topk_overlap_critique_effect_error_action
```

---

## 4. Teacher 对 student 已选 token 的概率

对 student 实际生成 token \(y_{i,t}\)，记录：

\[
p_{T_0}(y_{i,t})
\]

和：

\[
p_{T_c}(y_{i,t})
\]

### 4.1 W&B 指标

#### 截断轨迹

```text
teacher/student_token_prob_base_prefix_to_error
teacher/student_token_prob_critique_prefix_to_error
teacher/student_token_prob_critique_minus_base_prefix_to_error
```

其中：

\[
\Delta p
=
p_{T_c}(y_t)-p_{T_0}(y_t)
\]

#### 仅错误 action

```text
teacher/student_token_prob_base_error_action
teacher/student_token_prob_critique_error_action
teacher/student_token_prob_critique_minus_base_error_action
```

如果 critique teacher 更反对 student 的错误 token，通常预期：

```text
student_token_prob_critique_error_action
<
student_token_prob_base_error_action
```

### 4.2 同时保留 log-prob

为了避免极小概率被压缩，建议同时记录相同范围的平均 log-prob：

```text
teacher/student_token_logprob_base_prefix_to_error
teacher/student_token_logprob_critique_prefix_to_error
teacher/student_token_logprob_critique_minus_base_prefix_to_error

teacher/student_token_logprob_base_error_action
teacher/student_token_logprob_critique_error_action
teacher/student_token_logprob_critique_minus_base_error_action
```

概率曲线便于直观阅读，log-prob 曲线更适合分析小概率 token。

---

## 5. Teacher entropy

如果能取得完整 teacher logits，计算精确熵：

\[
H(\pi_T)
=
-\sum_v \pi_T(v)\log \pi_T(v)
\]

分别记录普通 teacher 和 critique teacher。

如果只能取得 top-k 概率，则先在各自 top-k support 内重新归一化，再计算 top-k 条件熵。该指标不是完整词表上的精确 entropy，名称必须带 `_topk_normalized`。

### 5.1 精确熵指标

```text
teacher/entropy_base_prefix_to_error
teacher/entropy_critique_prefix_to_error
teacher/entropy_base_error_action
teacher/entropy_critique_error_action
```

### 5.2 Top-k 归一化熵指标

```text
teacher/entropy_topk_normalized_base_prefix_to_error
teacher/entropy_topk_normalized_critique_prefix_to_error
teacher/entropy_topk_normalized_base_error_action
teacher/entropy_topk_normalized_critique_error_action
```

同样按照“逐轨迹平均，再对轨迹平均”聚合。

---

## 6. 每个训练 step 的时间

记录：

```text
time/student_rollout_sec
time/teacher_error_analysis_sec
time/critique_teacher_scoring_sec
time/base_teacher_scoring_sec
time/train_step_total_sec
time/evaluation_sec
```

具体定义：

- `time/student_rollout_sec`：student 与 ALFWorld 环境交互并生成完整 batch 轨迹的时间。
- `time/teacher_error_analysis_sec`：teacher 阅读失败轨迹、生成 critique，并完成结构化错误分析的时间。
- `time/critique_teacher_scoring_sec`：带 critique 特权信息的 teacher 对截断 student 轨迹计算概率的时间。
- `time/base_teacher_scoring_sec`：不带 critique 的普通 teacher 对同一截断 student 轨迹计算概率的时间。
- `time/train_step_total_sec`：当前 `global_step` 从开始 rollout 到本次参数更新完成的总时间。
- `time/evaluation_sec`：一次完整训练集或测试集评估所用时间，仅在执行评估时记录。

评估时间不要计入普通训练 step 的分项时间；如果训练主循环本身会阻塞等待评估，则仍应单独记录 `time/evaluation_sec`。

---

## 7. 当前训练 step 的轨迹准确率

这是当前训练 batch 在线 rollout 的成功率，不是完整训练集评测。

```text
train_step/trajectory_accuracy
```

计算：

\[
	ext{trajectory accuracy}
=
rac{
	ext{当前 batch 中成功完成任务的轨迹数}
}{
	ext{当前 batch 中总轨迹数}
}
\]

成功标准使用 ALFWorld 的：

```python
won == True
```

或等价的正环境奖励。

当前配置 `rollout.n=1` 时，它等价于当前 batch 的任务成功率。成功轨迹即使不参与 OPD，也必须进入该准确率的分母和分子。

同时记录：

```text

train_step/error_step_mean
critique/parse_failure_ratio
```

其中：

\[
\text{error step mean}
=
\frac{1}{N_{\text{valid critique}}}
\sum_i (\text{error_step}_i + 1)
\]

使用一基步骤编号，因此 teacher 判断第一步出错时记为 1。只统计 critique 成功解析且错误步骤未越界的失败轨迹。

\[
\text{parse failure ratio}
=
\frac{\text{critique 解析失败或错误步骤越界的失败轨迹数}}
{\text{调用 teacher 进行错误分析的失败轨迹总数}}
\]

缺少必要标签、字段为空、数字格式错误以及错误步骤越界，都计入 parse failure。低置信度拒绝不要计入 parse failure

---

## 8. 每个问题执行步数

这里的“步数”指实际执行的 environment action 数量，不是 token 数量。

对当前训练 batch 中每条轨迹：

```python
num_steps = len(trajectory_steps)
```

记录：

```text
rollout/steps_min
rollout/steps_max
rollout/steps_mean
```
额外区分成功和失败轨迹：

```text
rollout/success_steps_mean
rollout/failed_steps_mean
```

若某组没有有效轨迹，不记录该组平均值，或者记录 NaN；不要伪造为 0。

---

## 9. 训练基本指标

优先复用 veRL 已有指标名称；如果已有同名指标，不要重复计算。

### 9.1 OPD loss

```text
train/opd_loss
```

记录当前真正用于反向传播的 distillation loss，经过 response mask 和 loss aggregation 后的标量。

也建议记录：

```text
train/opd_valid_token_count
train/opd_valid_trajectory_count
```

便于判断 loss 波动是否来自有效样本数量变化。

### 9.2 Actor entropy

```text
actor/entropy
```

使用 veRL actor forward 已计算的 entropy，在有效 student action token 上聚合。

### 9.3 PPO KL

```text
actor/ppo_kl
```

用于观察参数更新前后 policy 的变化：

\[
	ext{approx PPO KL}
=
\operatorname{mean}
\left[
\log\pi_{	ext{old}}(y_t)
-
\log\pi_{	ext{new}}(y_t)

ight]
\]

即使当前使用直接蒸馏 loss、没有把 PPO KL 加进 loss，也可以把它作为更新幅度的诊断指标。

### 9.4 Gradient norm

```text
actor/grad_norm
```

记录 optimizer step 前的全局梯度范数，优先使用 veRL 已有值。

### 9.5 Learning rate

```text
actor/lr
```

直接记录当前 optimizer learning rate。

---

## 10. 训练集和测试集准确率

这两项是周期性评测，不是当前训练 batch 的在线成功率。

### 10.1 训练集评测准确率

```text
eval/train_accuracy
```

在固定的训练集评测子集上运行，不做参数更新：

\[
	ext{train accuracy}
=
rac{	ext{成功任务数}}{	ext{评测任务总数}}
\]

### 10.2 测试集评测准确率

```text
eval/test_accuracy
```

在固定的 held-out 测试集或 ALFWorld 指定 eval split 上运行：

\[
	ext{test accuracy}
=
rac{	ext{成功任务数}}{	ext{评测任务总数}}
\]

训练集和测试集评测必须使用完全相同的：

```text
temperature
top_p
max_steps
max_action_tokens
rollout.n
seed 规则
```

否则两条准确率曲线不可直接比较。

同时记录评测任务数：

```text
eval/train_num_tasks
eval/test_num_tasks
```

---

## 11. W&B 分组建议

### Distribution Comparison

```text
distribution/js_student_base_prefix_to_error
distribution/js_student_critique_prefix_to_error
distribution/js_critique_effect_prefix_to_error
distribution/js_student_base_error_action
distribution/js_student_critique_error_action
distribution/js_critique_effect_error_action

distribution/topk_overlap_student_base_prefix_to_error
distribution/topk_overlap_student_critique_prefix_to_error
distribution/topk_overlap_critique_effect_prefix_to_error
distribution/topk_overlap_student_base_error_action
distribution/topk_overlap_student_critique_error_action
distribution/topk_overlap_critique_effect_error_action
```

### Selected Student Token

```text
teacher/student_token_prob_base_prefix_to_error
teacher/student_token_prob_critique_prefix_to_error
teacher/student_token_prob_critique_minus_base_prefix_to_error
teacher/student_token_prob_base_error_action
teacher/student_token_prob_critique_error_action
teacher/student_token_prob_critique_minus_base_error_action
```

### Training

```text
train/opd_loss
train/opd_valid_token_count
train/opd_valid_trajectory_count
actor/entropy
actor/ppo_kl
actor/grad_norm
actor/lr
```

### Rollout

```text
train_step/trajectory_accuracy
train_step/success_trajectory_count
train_step/failed_trajectory_count
train_step/error_step_mean
critique/parse_failure_ratio
rollout/steps_min
rollout/steps_max
rollout/steps_mean
rollout/success_steps_mean
rollout/failed_steps_mean
```

### Timing

```text
time/student_rollout_sec
time/teacher_error_analysis_sec
time/critique_teacher_scoring_sec
time/base_teacher_scoring_sec
time/train_step_total_sec
time/evaluation_sec
```

### Evaluation

```text
eval/train_accuracy
eval/test_accuracy
eval/train_num_tasks
eval/test_num_tasks
```

---

## 12. 实现要求

1. 所有曲线使用同一个 `global_step`。
2. Distribution、overlap、selected-token probability 和 teacher entropy 必须同时提供：
   - `prefix_to_error`
   - `error_action`
   JS 与 overlap 的 support 必须分别使用 `student ∪ base teacher` 和 `student ∪ critique teacher` 的 pairwise top-k 并集，并在各自并集内重新归一化。
3. 上述分布类指标必须先逐轨迹平均，再对有效轨迹平均。
4. observation prompt token 只作为上下文，不能参与指标。
5. 成功轨迹进入在线准确率，但不进入错误步骤相关指标。
6. critique parse 失败或错误步骤越界的轨迹，不进入错误步骤相关指标。
7. 所有指标计算必须使用 `torch.no_grad()`，并在写入 W&B 前 `.detach()`。
8. 不要为了记录指标改变训练 loss 或梯度。
9. 指标为空时不要伪造正常数值；记录有效样本数，并跳过该指标或记录 NaN。
10. W&B 当前需要在启动脚本中启用，例如将：

```bash
trainer.logger='[console]'
```

改为项目实际支持的 console + wandb logger 配置。
