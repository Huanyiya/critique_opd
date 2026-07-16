一、OPID 如何启动环境 actor

OPID 的调用链是：

主进程
  └── 创建一个 TaskRunner Ray actor
        └── TaskRunner.run()
              └── make_envs(config) 只调用一次
                    ├── 创建 train AlfworldEnvs
                    │     └── 128 个 AlfworldWorker Ray actor
                    └── 创建 val AlfworldEnvs
                          └── 128 个 AlfworldWorker Ray actor
1. 只有一个 TaskRunner 创建环境

OPID 主进程初始化 Ray 后，只创建一个 TaskRunner：

runner = TaskRunner.remote()
ray.get(runner.run.remote(config))

然后在唯一的 TaskRunner.run() 中执行一次：

envs, val_envs = make_envs(config)

环境不是在模型 rollout worker、FSDP worker 或其他并发 worker 中创建的。

所以 OPID 中实际只有：

一个 train 环境管理对象；
一个 val 环境管理对象；
一个统一的环境 actor 创建者和句柄持有者。
2. 启动时一次性创建 train 和 val actor

OPID 脚本配置：

TRAIN_DATA_SIZE = 16
VAL_DATA_SIZE   = 128
GROUP_SIZE      = 8

make_envs() 创建：

train env_num = train_batch_size = 16
train group_n = 8

val env_num = val_batch_size = 128
val group_n = 1

因此启动时创建：

16×8=128

个 train actor，以及：

128×1=128

个 val actor，总共 256 个环境 actor。两套环境都在 make_envs() 中集中创建。

3. 一个中央 AlfworldEnvs 保存全部 actor handle

AlfworldEnvs 是 TaskRunner 进程里的普通 Python 对象，不是 Ray actor。它集中保存：

self.workers = []

然后顺序创建每一个 Ray actor：

for i in range(self.num_processes):
    worker = env_worker.remote(
        config,
        seed + (i // self.group_n),
        base_env,
    )
    self.workers.append(worker)

所以 ownership 很明确：

唯一 TaskRunner
  └── 唯一 AlfworldEnvs
        └── workers[0 ... N-1]

没有多个 manager 竞争同一个 actor。

4. 同组 actor 使用相同 seed

actor i 的 seed 是：

seed + i // group_n

当 group_n=8 时：

actor 0~7   -> seed 0
actor 8~15  -> seed 1
actor 16~23 -> seed 2
...

因此每组 8 个独立环境拥有相同的游戏顺序，reset 时进入同一个任务，但各自可以产生独立 student rollout。这正是你想复刻的 OPID group 行为。

5. reset 和 step 是中央批量调度

OPID 每次 reset：

for worker in self.workers:
    futures.append(worker.reset.remote())

results = ray.get(futures)

每次 step 也是：

for i, worker in enumerate(self.workers):
    futures.append(worker.step.remote(actions[i]))

results = ray.get(futures)

即一个中央 vector environment 同时管理全部环境 actor。

二、OPID 如何关闭环境 actor
1. OPID 定义了完整的关闭链

上层：

EnvironmentManagerBase.close()
    -> self.envs.close()

ALFWorld 底层：

AlfworldEnvs.close()
    -> 遍历 self.workers
    -> ray.kill(worker)

因此从设计上，它可以由唯一 manager 一次关闭整个环境池：

envs.close()
  └── kill 全部 train workers

val_envs.close()
  └── kill 全部 val workers

不存在“一个 manager 只知道一部分 actor”的问题。

真正写成 OPID 方式应该怎么改

目标架构应该是：

TaskRunner / AgentLoopManager
  └── 创建唯一 ALFWorldEnvPoolManager Ray actor
        ├── 持有全部 train env actors
        └── 持有全部 val env actors

12 个 AgentLoopWorker
  └── 每个只保存同一个 manager actor handle
        └── 不创建本地 manager

具体需要：

删除 AgentLoopWorker.__init__() 中的：
initialize_alfworld_env_manager_from_config(config)
不再依赖进程本地：
_ENV_MANAGER_CACHE
在唯一的 driver/TaskRunner/AgentLoopManager 初始化阶段创建一次环境 manager。
把同一个 manager actor handle 传给全部 AgentLoopWorker。
manager 内部集中创建和保存全部 environment actor handles。
AgentLoopWorker 根据：
worker_id = sample_index * group_size + session_id

向中央 manager 获取对应 actor，或让 manager 代理 reset/step。

在 TaskRunner 使用：
manager = None
try:
    manager = ...
    trainer.fit()
finally:
    if manager is not None:
        ray.get(manager.close.remote())
        ray.kill(manager, no_restart=True)
manager 的 close() 统一：
for worker in all_workers:
    worker.close.remote()
for worker in all_workers:
    ray.kill(worker)

这样才是：

OPID 的中央创建和中央 ownership，加上比 OPID 更可靠的显式关闭。

当前实现不需要推翻环境 worker、seed 和 sample/session 映射；主要应重构的是 manager 的创建位置和所有权。