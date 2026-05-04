# Office_Products GRPO 训练复盘

本文档用于记录 MiniOneRec 在 Office_Products 上从 SFT 模型继续做 GRPO 的完整过程，方便后续实验复盘、论文/项目汇报和面试讲解。

## 1. 项目背景

MiniOneRec 是一个基于生成式大模型的推荐系统项目。核心思路是把商品映射成结构化的 Semantic ID，然后让语言模型根据用户历史交互序列生成下一个商品的 Semantic ID。

本次实验的任务是 Office_Products 类目上的下一物品推荐：

- 数据集：Office_Products
- SID 方法：rqkmeans_plus
- 基座模型：Qwen2.5-7B
- 已完成阶段：SID 构建、SFT
- 本次目标：从效果最好的 SFT LoRA checkpoint 出发，继续做 GRPO 强化学习，并评估是否超过 SFT baseline

本次使用的 SFT checkpoint：

```text
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/sft/new_lora/final_checkpoint
```

该 checkpoint 是当前 Office_Products 上表现最好的 SFT 版本，后续所有 GRPO 实验都以它作为起点或对比基线。

## 2. 环境与资源

训练环境：

```text
/mnt/bms_afs/miniconda3/envs/minionerec312
```

主要硬件资源：

```bash
srun -p gpu --gres=gpu:nvidia_h200:4 --pty bash
```

训练使用 4 张 H200 GPU，通过 `accelerate launch` 启动多卡训练。

主要脚本：

```text
sbatch_office_grpo_resume.sh
grpo_h200_7b.sh
rl.py
minionerec_trainer.py
evaluate.sh
```

实际 GRPO 调用链：

```text
sbatch_office_grpo_resume.sh
  -> grpo_h200_7b.sh full
    -> accelerate launch rl.py
      -> trl.GRPOConfig
      -> ReReTrainer
```

其中 `rl.py` 是训练入口，`minionerec_trainer.py` 里的 `ReReTrainer` 是项目自定义的 GRPO trainer。

## 3. 训练输入与输出

Office_Products 相关路径：

```text
train_file:
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/data/train/Office_Products_5_2016-10-2018-11.csv

eval_file:
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/data/valid/Office_Products_5_2016-10-2018-11.csv

test_file:
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/data/test/Office_Products_5_2016-10-2018-11.csv

info_file:
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/data/info/Office_Products_5_2016-10-2018-11.txt

sid_index_path:
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/sid_office/rqkmeans_plus/Office_Products.index.json

item_meta_path:
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/data/Amazon/index/Office_Products.item.json
```

第一次 GRPO 输出目录：

```text
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/grpo/new_lora
```

中断后继续训练输出目录：

```text
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/grpo/new_lora_continue_from_522
```

最终评估结果路径：

```text
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/results/office_rqkmeans_plus/grpo/new_lora_continue_from_522/final_result_Office_Products.json
```

## 4. GRPO 训练配置

Office_Products 的关键训练参数如下：

```bash
CATEGORY=Office_Products
MODEL_PATH=output/office_rqkmeans_plus/sft/new_lora/final_checkpoint
OUTPUT_DIR=output/office_rqkmeans_plus/grpo/new_lora
NPROC=4
TRAIN_BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=16
NUM_GENERATIONS=4
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=5e-6
BETA=1e-3
REWARD_TYPE=ranking
OPTIM=paged_adamw_32bit
```

`rl.py` 里使用的 `GRPOConfig` 关键参数：

```python
GRPOConfig(
    max_completion_length=128,
    num_generations=num_generations,
    temperature=temperature,
    per_device_train_batch_size=train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    beta=beta,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
    num_train_epochs=num_train_epochs,
    bf16=True,
    optim=optim,
    lr_scheduler_type="cosine",
    save_strategy="steps",
    save_steps=0.1,
)
```

训练数据由多个任务拼接得到：

- `SidDataset`：用户历史 SID 到目标 SID 的推荐任务
- `RLTitle2SidDataset`：商品 title 到 SID 的对齐任务
- `RLSeqTitle2SidDataset`：序列 title 到 SID 的辅助任务

这样做的目的不是只优化推荐命中率，还要保持模型对商品标题和 SID 空间的理解，降低生成非法 SID 或语义漂移的风险。

## 5. Reward 设计

本次使用：

```bash
REWARD_TYPE=ranking
```

在 `rl.py` 中对应两个 reward：

```python
reward_fun = [rule_reward, ndcg_rule_reward]
```

`rule_reward`：

- 如果生成的 SID 与目标 SID 完全一致，reward = 1
- 否则 reward = 0

`ndcg_rule_reward`：

- 同一个 prompt 生成 `num_generations` 个候选
- 如果其中存在目标 SID，则根据目标 SID 在候选中的位置给 NDCG 风格奖励
- 越靠前的候选奖励越高
- 如果一组候选都没有命中，则该组 reward 为 0

这个 reward 设计的动机是：推荐评估通常看 HR@K 和 NDCG@K，而不是只看 top-1 完全匹配。因此 GRPO 阶段希望把模型从 SFT 的 token-level imitation，进一步推向 ranking metric。

## 6. 实际训练过程

### 6.1 Smoke Test

先跑了小规模 smoke test，确认：

- 训练脚本可以启动
- 数据路径正确
- reward function 可以正常计算
- 多卡 accelerate 配置可用
- checkpoint 可以保存

smoke test 成功后，开始 full GRPO。

### 6.2 Full GRPO

第一次 full GRPO 从 SFT `new_lora/final_checkpoint` 启动，输出到：

```text
output/office_rqkmeans_plus/grpo/new_lora
```

该 run 训练到了：

```text
checkpoint-522
```

之后任务中断或被取消，没有完成到最终 checkpoint。

保留下来的主要 checkpoint：

```text
checkpoint-87
checkpoint-174
checkpoint-261
checkpoint-348
checkpoint-435
checkpoint-522
```

### 6.3 Resume 问题

为了从 `checkpoint-522` 继续训练，后来给代码增加了 resume 参数：

```python
resume_from_checkpoint: str = ""
trainer.train(resume_from_checkpoint=resume_from_checkpoint or None)
```

同时在 shell 脚本里增加：

```bash
RESUME_FROM_CHECKPOINT
--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"
```

但是严格 resume 失败，原因是当前环境里的 PyTorch 是 2.5.1，而 Transformers 在恢复 optimizer/scheduler 状态时触发安全检查，需要 `torch>=2.6` 才能安全加载某些 `torch.load` checkpoint 内容。

因此最终采用了折中方案：

```text
把 MODEL_PATH 指向 checkpoint-522 的 LoRA 权重，
重新启动一个新的 GRPO run，
输出到 new_lora_continue_from_522。
```

这不是严格的 Trainer resume。它只恢复了模型权重，没有恢复：

- optimizer state
- lr scheduler state
- dataloader state
- RNG state
- global step 对应的训练动态

因此后续结果需要谨慎解读。

### 6.4 Continue From 522

继续训练 run 的输入模型：

```text
output/office_rqkmeans_plus/grpo/new_lora/checkpoint-522
```

继续训练输出：

```text
output/office_rqkmeans_plus/grpo/new_lora_continue_from_522
```

最终 checkpoint：

```text
output/office_rqkmeans_plus/grpo/new_lora_continue_from_522/final_checkpoint
```

续训过程中观察到两个重要现象：

- 新 run 的 learning rate 从 0 重新 warmup，说明 scheduler 不是从全局 step 522 继续
- 续训末尾出现过明显 KL spike，说明训练稳定性存在风险

这意味着中断后的非严格续训可能对最终性能产生负面影响。

## 7. 评估结果

SFT `new_lora` baseline：

```text
HR@1  = 0.0917
HR@3  = 0.1270
HR@5  = 0.1463
HR@10 = 0.1697

NDCG@1  = 0.0917
NDCG@3  = 0.1125
NDCG@5  = 0.1204
NDCG@10 = 0.1279

invalid = 0
```

GRPO `new_lora_continue_from_522/final_checkpoint`：

```text
HR@1  = 0.08364159
HR@3  = 0.11590629
HR@5  = 0.13173037
HR@10 = 0.15782984

NDCG@1  = 0.08364159
NDCG@3  = 0.10270681
NDCG@5  = 0.10917064
NDCG@10 = 0.11759754

invalid = 0
```

对比结论：

- GRPO final 没有超过 SFT baseline
- HR 和 NDCG 全部下降
- invalid 仍然为 0，说明约束解码和 SID 合法性没有明显问题
- 当前最优模型仍是 SFT `new_lora/final_checkpoint`

## 8. 结果分析

这次 GRPO 没有提升，可能有两类原因。

第一类是训练恢复问题。严格 resume 没有成功，后续采用的是从 `checkpoint-522` 权重重新开一个 run。对于 GRPO 这类 on-policy 或近似 on-policy 的强化学习训练，optimizer、scheduler、采样随机性和 KL 约束都比较敏感。学习率重新 warmup 和 KL spike 都说明训练轨迹发生了不连续。

第二类是 GRPO 配置本身可能不适合当前 SFT 模型。SFT `new_lora` 已经较强，GRPO 的 sparse exact-match reward 可能信号不足；同时 `learning_rate=5e-6`、`beta=1e-3`、`num_generations=4` 的组合可能导致策略更新过强或 reward 方差较高。最终虽然没有生成非法 SID，但排序质量下降，说明模型可能被 RL 更新推离了 SFT 的较优局部区域。

因此不能简单断言“性能下降一定是中断导致”。更严谨的结论是：中断后的非严格续训是一个高风险因素，但需要评估中断前的 checkpoint 才能判断主因。

## 9. 后续验证建议

最优先要做的是评估中断前的 checkpoint：

```text
output/office_rqkmeans_plus/grpo/new_lora/checkpoint-87
output/office_rqkmeans_plus/grpo/new_lora/checkpoint-174
output/office_rqkmeans_plus/grpo/new_lora/checkpoint-261
output/office_rqkmeans_plus/grpo/new_lora/checkpoint-348
output/office_rqkmeans_plus/grpo/new_lora/checkpoint-435
output/office_rqkmeans_plus/grpo/new_lora/checkpoint-522
```

判断逻辑：

- 如果 `checkpoint-522` 已经低于 SFT，说明主要问题是 GRPO 配置或 reward 设计
- 如果 `checkpoint-522` 接近或高于 SFT，但 `continue_from_522/final_checkpoint` 下降，说明非严格续训影响很大
- 如果早期 checkpoint 曾超过 SFT，说明 GRPO 有收益，但需要 early stopping

下一轮实验可以尝试：

- 降低 learning rate，例如 `1e-6` 或 `2e-6`
- 增大 KL 约束，例如提高 `beta`
- 减少训练步数并加入 checkpoint-level evaluation
- 提高 `num_generations`，降低 ranking reward 方差
- 只从 SFT 重新完整跑一版短 GRPO，不使用中断续训结果
- 对 reward 做更平滑的设计，避免只有 exact-match 命中才有有效信号

## 10. 面试表达版本

可以这样介绍这个实验：

> 我在 MiniOneRec 里做的是生成式推荐。系统先把商品通过 rqkmeans_plus 映射成 Semantic ID，然后用 Qwen2.5-7B 学习根据用户历史生成下一个商品的 Semantic ID。SFT 之后，我尝试用 GRPO 进一步直接优化推荐排序指标。

> GRPO 阶段我没有只用 token-level loss，而是设计了和推荐指标对齐的 reward：一个 exact-match reward 判断是否生成目标 SID，另一个 NDCG-style reward 根据目标 SID 在多候选生成里的排序位置给奖励。这样目标更接近 HR@K 和 NDCG@K。

> 实验中我先做 smoke test 验证多卡、数据、reward 和保存逻辑，再用 4 张 H200 跑 full GRPO。训练在 checkpoint-522 后中断，严格 resume 由于 PyTorch/Transformers 的 checkpoint 安全加载限制失败，所以我采用了从 checkpoint-522 权重重新启动的折中方案。

> 最终 GRPO 没有超过 SFT，HR@10 从 0.1697 降到 0.1578，NDCG@10 从 0.1279 降到 0.1176。我的判断是两个因素叠加：一是非严格 resume 导致 optimizer/scheduler/RNG 状态不连续，二是当前 reward 和超参可能对已经较强的 SFT 模型更新过激。后续我会优先评估中断前各 checkpoint，判断是 GRPO 本身退化还是续训导致退化。

这个回答能体现三个点：

- 能把推荐任务转成生成式建模问题
- 能解释为什么 GRPO reward 要对齐 HR/NDCG
- 能客观分析实验失败原因，而不是只报结果

## 11. 当前结论

当前 Office_Products 最好模型仍是：

```text
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/sft/new_lora/final_checkpoint
```

GRPO final checkpoint 可作为复盘材料，但暂时不建议作为主模型使用：

```text
/mnt/bms_afs/users/yuedongxu/GAO/MiniOneRec/output/office_rqkmeans_plus/grpo/new_lora_continue_from_522/final_checkpoint
```

下一步最有价值的工作是评估 GRPO 中断前所有 checkpoint，确认是否存在 early checkpoint 超过 SFT。如果有，就用 early stopping 选最佳 GRPO；如果没有，就需要重新设计 reward 或降低 RL 更新强度。
