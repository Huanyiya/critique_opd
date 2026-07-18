Codex 的解释大体正确，但有一个关键点：

> **Semaphore 的“执行槽位”不是 GPU 槽位，也不代表已经开始在 GPU 上计算。它只是允许这条请求继续往下发送。**

## 先分清四个概念

### 1. `12 个 AgentLoopWorker`

它们是 **CPU 调度进程**，不对应 12 张 GPU。

当前 96 条轨迹被切给 12 个 worker，每个大约拿 8 条；每个 worker 又把自己的 8 条轨迹作为异步任务推进。

### 2. Teacher replica 数

假设：

```text
4 张 teacher GPU
TEACHER_TP=1
```

就是 4 个独立 teacher replica。

如果：

```text
4 张 teacher GPU
TEACHER_TP=2
```

则是 2 个 replica。

所以并发主要参考 **replica 数**，不是单纯参考 GPU 张数。

### 3. Semaphore 容量

```python
Semaphore(4)
```

表示最多允许 4 条请求越过这一道门。

第 5 条会停在：

```python
await semaphore.acquire()
```

此时它还没有发给 teacher vLLM。

### 4. vLLM 自己的队列

请求获得 semaphore 后，会调用：

```text
AgentLoopWorker
→ teacher manager
→ 负载均衡器
→ 某个 teacher vLLM replica
→ vLLM scheduler
```

vLLM 内部仍然有：

```text
waiting requests
running requests
prefill
decode
KV cache
```

所以答案是：

> **获得 semaphore 槽位后，仍然可能排队等待 GPU。**

---

# 一条失败轨迹的完整流程

```text
AgentLoopWorker 中的一条 trajectory
        │
        ├─ student_action_start
        │      ↓
        │   请求 student vLLM
        │      ↓
        │   vLLM 内部排队/生成
        │      ↓
        │   environment.step
        │
        ├─ 重复最多 30 步
        │
        ├─ 轨迹失败
        │
        ├─ teacher_critique_queued
        │      ↓
        │   等 semaphore
        │
        ├─ teacher_critique_start
        │      ↓
        │   请求已经发给 teacher manager
        │      ↓
        │   仍可能在负载均衡器/vLLM 中等待
        │      ↓
        ├─ teacher_critique_end
        │
        ├─ 解析 error_step
        │
        ├─ privileged teacher logprob
        │
        └─ unprivileged teacher logprob
```

当前 critique 之后，代码会继续计算 critique-conditioned 和普通 teacher 概率。

---

# 实际上有三层排队

假设全局 critique semaphore 是 4：

### 第一层：Semaphore 前面

```text
68 条失败轨迹
其中 4 条获得 semaphore
剩余 64 条停在 queued
```

这 64 条还没发给 teacher。

### 第二层：Teacher manager / 负载均衡

获得槽位的 4 条请求，需要选择发送给哪个 replica。

理想情况下：

```text
request 1 → replica 0
request 2 → replica 1
request 3 → replica 2
request 4 → replica 3
```

但如果负载均衡不均、某个 replica 慢或还有 logprob 请求，仍可能等待。

### 第三层：vLLM scheduler

请求已经到达某个 replica，但该 replica 可能正在：

* 给别的请求做长 prompt prefill；
* decode critique；
* 计算 logprob；
* 等 KV cache；
* 发生 preemption。

所以日志含义是：

```text
queued 很久，没有 start
→ 卡在 semaphore 前

start 很久，没有 end
→ 已发送请求，卡在 teacher manager / vLLM 内部
```

---

# 三个阶段分别需要多少并发

## 一、Student rollout

Student rollout 需要**高并发**。

因为一条 ALFWorld 轨迹中会交替发生：

```text
student 生成
→ CPU environment.step
→ Python 构造 prompt
→ student 再生成
```

GPU 并不是持续工作。需要很多轨迹交错推进，让一条轨迹在等环境时，其他轨迹给 GPU 提供请求。

### 你的配置

```text
4 个 student replica
12 个 AgentLoopWorker
每个约 8 条 trajectory
总活跃轨迹最多约 96
```

这不一定过多。对于交互式 agent rollout，活跃轨迹数明显大于 GPU replica 数通常是合理的。

建议初始保持：

```text
AgentLoopWorker 数：12
活跃 student trajectory：最多约 96
```

但把单步输出限制为：

```bash
ALFWORLD_MAX_ACTION_TOKENS=128
```

如果日志显示：

```text
student_action_start 后大量请求几分钟不结束
vLLM Waiting 很高
频繁 preemption
```

再考虑降为：

```text
8 个 AgentLoopWorker
```

或者每个 worker 同时只推进 4 条，而不是 8 条。

### Student 并发不应等于 GPU 数

4 张 student GPU 只运行 4 条轨迹会很浪费，因为其中任何一条在执行环境时，对应 GPU 就缺请求。

因此 student 通常需要：

```text
活跃轨迹数 ≫ replica 数
```

---

## 二、Teacher critique

Critique 是：

```text
约 18k token 长 prompt prefill
+ 生成最多 192/512 token
```

它不像 student rollout，没有环境等待，而且单条请求很重。

所以 critique 不需要 96 条并发。

对于：

```text
4 张 teacher GPU
TP=1
4 个 teacher replica
```

最安全的初始值是：

```text
全局 critique 并发 = 4
```

相当于先尽量让每个 replica 同时处理一条长请求。

之后根据日志调整：

```text
GPU 利用率低、KV cache 很空、无 preemption
→ 可以试 8

显存很高、频繁 preemption、单请求延迟暴涨
→ 保持 4，甚至降到 2
```

你的 prompt 很长，而且之前出现过 OOM/超长等待，所以**先用 4，不建议直接用 8**。

---

## 三、Teacher logprob

Teacher logprob 与 critique 不一样：

```text
不自回归生成新 token
只做 prompt + student response 的前向概率计算
```

理论上比 critique 更适合批处理。

但你的实现会把错误步骤以前的多个 step 展开，然后分别计算：

```text
privileged logprob
unprivileged logprob
```

当前这些调用分散在不同 worker 中。

例如一条轨迹有 5 个训练步骤，就可能是：

```text
5 次 privileged
+ 5 次 unprivileged
= 10 次 scoring 请求
```

如果大量 worker 同时提交，也可能挤爆 teacher。
