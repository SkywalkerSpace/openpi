# 梳理 openpi 代码

> /Users/skywalker/code/experiment/openpi/src/openpi, 读代码，梳理功能，结构，操作方法等等

<details><summary>1 previous message</summary>

> 我先把 `/Users/skywalker/code/experiment/openpi/src/openpi` 这块代码的目录结构和入口摸清楚，再顺着核心模块看它能做什么、怎么用。之后我会把功能、结构和操作方式整理成一份清晰的中文梳理。
</details>

我已经把目录扫出来了，下一步会抓主入口、模型/策略/训练/服务这几条线的核心文件，先建立整体地图，再补细节。这样能避免只看零散实现而漏掉它的“使用方式”。

> 继续

<details><summary>6 previous messages</summary>

> 我继续往下拆核心实现：模型定义、tokenizer、transforms、训练数据和几个具体 policy。这样能把“这个库到底怎么拼起来跑”的链路补完整。
>
> <details><summary>Explored 5 files</summary>
>
> - Read `./openpi/src/openpi/models/tokenizer.py`
> - Read `./openpi/src/openpi/models/pi0.py`
> - Read `./openpi/src/openpi/models/pi0_config.py`
> - Read `./openpi/src/openpi/transforms.py`
> - Read `./openpi/src/openpi/models/pi0_fast.py`
>
> </details>
>
> 我已经看到主干了：`model.py` 定义统一数据结构，`pi0/pi0_fast` 是两条模型线，`transforms.py` 负责把原始数据变成模型需要的格式。现在补上训练、数据加载和具体 policy，最后我会把整条调用链串起来。
>
> <details><summary>Explored 6 files</summary>
>
> - Read `./openpi/src/openpi/policies/droid_policy.py`
> - Read `./openpi/src/openpi/training/weight_loaders.py`
> - Read `./openpi/src/openpi/transforms.py`
> - Read `./openpi/src/openpi/training/data_loader.py`
> - Read `./openpi/src/openpi/policies/aloha_policy.py`
> - Read `./openpi/src/openpi/policies/libero_policy.py`
>
> </details>
>
> 我已经把主要“业务件”看完了。最后再查一下训练配置的入口和现成测试/示例，确认这个包到底提供了哪些可直接执行的操作路径。
>
> <details><summary>Explored 5 files, a search</summary>
>
> - Read `./openpi/src/openpi/training/config.py`
> - Read `./openpi/src/openpi/models/__init__.py`
> - Read `./openpi/src/openpi/training/checkpoints.py`
> - Read `./openpi/src/openpi/policies/policy_config.py`
> - Read `./openpi/src/openpi/models/model_test.py`
> - Searched for `class TrainConfig|_CONFIGS|tyro|def main|if __name__ == '__main__'` in `training`
>
> </details>
</details>

我把 `openpi/src/openpi` 这套代码的主干梳理出来了，整体上它是一个“机器人模仿学习 / policy 训练与部署”框架，核心分成 5 层：

**1. 核心抽象层**
- [models/model.py](./openpi/src/openpi/models/model.py) 定义了统一的数据结构和模型接口。
- 关键对象是 `Observation` 和 `Actions`。
- `Observation` 统一了输入格式：
  - `images`：多路相机图像
  - `image_masks`：每路图像是否有效
  - `state`：机器人低维状态
  - `tokenized_prompt` / `tokenized_prompt_mask`：语言输入
  - `token_ar_mask` / `token_loss_mask`：FAST 模型专用
- `BaseModelConfig` 定义了模型配置的公共接口：
  - `create()`：实例化模型
  - `load()`：从参数恢复 JAX 模型
  - `load_pytorch()`：从 PyTorch 权重恢复
  - `inputs_spec()`：返回输入/输出 shape 规范
- `preprocess_observation()` 做统一预处理：
  - 图像 resize
  - 训练时增强
  - 默认补 image mask

**2. 模型层**
- 标准 Pi0： [models/pi0.py](./openpi/src/openpi/models/pi0.py)
- Pi0 配置： [models/pi0_config.py](./openpi/src/openpi/models/pi0_config.py)
- FAST 版本： [models/pi0_fast.py](./openpi/src/openpi/models/pi0_fast.py)

这两条线的差异很明显：

- **Pi0 / Pi05**
  - 输入是图像 + state + prompt
  - 输出是连续 action 序列
  - 训练目标是 flow matching / denoising 风格的回归
  - `sample_actions()` 用迭代式积分采样，把噪声一步步推成动作
  - `Pi05` 主要差别：
    - state 被离散化进语言 token
    - action expert 使用 adaRMSNorm 风格的时间条件

- **Pi0FAST**
  - 更像“把动作也 token 化”的自回归生成
  - `sample_actions()` 是逐 token decode
  - `compute_loss()` 是 token 级交叉熵
  - `FASTTokenizer` 负责把动作编码成 PaliGemma vocab 里的特定 token 区间

**3. Tokenizer / 文本动作编码**
- [models/tokenizer.py](./openpi/src/openpi/models/tokenizer.py)

这里有三类 tokenizer：

- `PaligemmaTokenizer`
  - 用于普通 Pi0 / Pi05
  - prompt 会被编码成固定长度 token 序列
  - Pi05 时还会把 state 离散化后拼进 prompt
- `FASTTokenizer`
  - 用于 FAST 模型
  - prompt + state + action 都会被组织成一个 token 序列
  - action 会映射到 PaliGemma vocab 尾部的专用 token 区间
  - 还能把生成出来的 token 反解回动作
- `BinningTokenizer` / `FSQTokenizer`
  - 是 RoboArena / baseline 相关的其他动作 token 化方式
  - 不属于 Pi0 主线，但保留了兼容实现

**4. 数据变换层**
- [transforms.py](./openpi/src/openpi/transforms.py)

这是整个库里非常关键的一层，负责把“原始数据”变成“模型输入”。

主要能力：

- `RepackTransform`
  - 把嵌套字段重组到统一结构
- `InjectDefaultPrompt`
  - 没有 prompt 时补默认任务描述
- `ResizeImages`
  - 图像 resize/pad 到 224x224
- `Normalize` / `Unnormalize`
  - 用离线统计量做标准化或 quantile 归一化
- `PadStatesAndActions`
  - 把 state / action 补到模型维度
- `TokenizePrompt`
  - 普通 Pi0 的文本 token 化
- `TokenizeFASTInputs`
  - FAST 模型输入 token 化
- `ExtractFASTActions`
  - 把 FAST 输出 token 还原成动作
- `PromptFromLeRobotTask`
  - 从 LeRobot dataset 的 task index 生成 prompt

还有一个重要设计是 `Group`：
- `Group.inputs`：训练/推理前的输入变换
- `Group.outputs`：推理后的输出变换
- `push()` 支持把一组变换前后追加进去

**5. 训练 / 数据 / checkpoint / 权重加载**
- 训练配置： [training/config.py](./openpi/src/openpi/training/config.py)
- 数据加载： [training/data_loader.py](./openpi/src/openpi/training/data_loader.py)
- checkpoint： [training/checkpoints.py](./openpi/src/openpi/training/checkpoints.py)
- 权重加载： [training/weight_loaders.py](./openpi/src/openpi/training/weight_loaders.py)

这里的思路是：

- `TrainConfig` 聚合：
  - `model`
  - `data`
  - `optimizer`
  - `assets_dirs`
  - `batch_size`
  - `num_workers`
  - `seed`
- `data_loader.py` 会根据 config 创建：
  - `LeRobot` 数据集
  - 或 `DROID RLDS` 数据集
  - 或 fake dataset
- 然后把数据依次过：
  - repack transforms
  - data transforms
  - normalize
  - model transforms
- `checkpoints.py` 负责：
  - 保存 train_state
  - 单独保存可推理参数 `params`
  - 保存 assets（比如 norm stats）
  - 恢复时重新合并参数
- `weight_loaders.py` 提供了几种权重入口：
  - `NoOpWeightLoader`
  - `CheckpointWeightLoader`
  - `PaliGemmaWeightLoader`

**6. Policy 和部署层**
- 通用 policy 封装： [policies/policy.py](./openpi/src/openpi/policies/policy.py)
- websocket 服务： [serving/websocket_policy_server.py](./openpi/src/openpi/serving/websocket_policy_server.py)
- 具体机器人 policy：
  - [policies/aloha_policy.py](./openpi/src/openpi/policies/aloha_policy.py)
  - [policies/droid_policy.py](./openpi/src/openpi/policies/droid_policy.py)
  - [policies/libero_policy.py](./openpi/src/openpi/policies/libero_policy.py)

`Policy` 的作用是把模型包成统一推理接口：

- 先做输入 transforms
- 再把数据转成 JAX 或 PyTorch 张量
- 调 `model.sample_actions`
- 再做输出 transforms
- 最终返回统一字典

`create_trained_policy()` 会：
- 自动判断 checkpoint 是 JAX 还是 PyTorch
- 恢复模型和 norm stats
- 拼接输入/输出 transforms
- 返回可直接 inference 的 policy 对象

`WebsocketPolicyServer` 则把 policy 暴露成 websocket 服务：
- 启动后先发 metadata
- 每次收到 obs 就返回 action
- 还带简单的 timing 信息
- `/healthz` 支持健康检查

**7. 具体机器人适配层**
这部分是“把各类机器人数据对接到统一接口”的适配器：

- Aloha：
  - [policies/aloha_policy.py](./openpi/src/openpi/policies/aloha_policy.py)
  - 会处理：
    - 图像通道重排
    - state / action 维度适配
    - 左右臂 joint flip
    - gripper 的角度/线性空间转换
- DROID：
  - [policies/droid_policy.py](./openpi/src/openpi/policies/droid_policy.py)
  - 会处理：
    - 两路图像
    - 7D state + gripper 拼接
    - 按模型类型构造不同的图像 key
- Libero：
  - [policies/libero_policy.py](./openpi/src/openpi/policies/libero_policy.py)
  - 逻辑类似，重点是把输入 key 和 action 维度对齐

**8. 代码里能看出的使用方式**
大概的典型流程是：

1. 选一个 `TrainConfig`
2. `data.create(...)` 生成数据管线
3. `model.create(...)` 生成模型
4. 用 data_loader 产出训练 batch
5. 训练时模型直接吃 `Observation + Actions`
6. 保存 checkpoint
7. 用 `create_trained_policy()` 恢复成推理 policy
8. 如果需要，交给 `WebsocketPolicyServer` 对外服务

**9. 从测试里能确认的行为**
- [models/model_test.py](./openpi/src/openpi/models/model_test.py) 说明：
  - `Pi0Config()` 默认能构造模型
  - `Pi0FASTConfig()` 默认也能构造
  - `compute_loss()` 和 `sample_actions()` 的输出 shape 是固定的
  - LoRA variant 也被支持
  - 模型恢复 checkpoint 的路径是通的

**10. 一句话概括**
这套 `openpi` 代码本质上是在做：
- 统一机器人输入输出格式
- 统一训练和推理数据管线
- 支持两类模型范式：
  - 连续动作生成的 Pi0 / Pi05
  - token 化动作生成的 Pi0FAST
- 再把它们包装成可训练、可恢复、可在线服务的 policy 系统

如果你愿意，我下一步可以继续帮你做其中一种更细的梳理：

1. 按“训练流程”画一条完整数据流
2. 按“推理流程”画一条完整调用链
3. 按“文件级别”逐个讲 `training/config.py` 里的各个 config 类
4. 直接帮你总结“怎么用这个库跑一个 policy / train 一个模型”


---

这个包的核心定位很明确：它不是一个通用机器人框架，而是围绕 openpi 的三类策略模型做的完整闭环，包含模型定义、数据变换、训练配置、推理封装和 websocket 部署。总的控制链路是：训练配置选型 -> 载入模型和归一化统计 -> 组装输入/输出 transforms -> 生成 policy -> 通过 websocket server 对外服务。

**结构梳理**
- src/openpi/models/model.py 定义了最底层的抽象：Observation、Actions、BaseModel、BaseModelConfig，以及参数恢复和观测预处理。
- src/openpi/models/pi0_config.py 和 src/openpi/models/pi0.py 是 π0 / π0.5 的实现，负责 flow matching 风格的动作生成。
- src/openpi/models/pi0_fast.py 和 src/openpi/models/tokenizer.py 是 π0-FAST 路线，核心区别是把动作离散化成 token，自回归解码。
- src/openpi/policies/policy.py 是运行时 policy 包装器，统一处理输入 transform、模型推理、输出 transform，还带有录制器。
- src/openpi/policies/policy_config.py 负责把训练配置、checkpoint、norm stats、transform 链拼成可直接推理的 policy。
- src/openpi/policies/aloha_policy.py、src/openpi/policies/droid_policy.py 是平台适配层，把不同机器人/数据集的字段映射到统一格式。
- src/openpi/training/config.py 是训练配置中心，里面注册了所有预置配置，比如 pi0_aloha、pi05_droid、pi05_libero、pi0_fast_droid 等。
- src/openpi/transforms.py 是整个数据管道的“积木库”，负责 repack、归一化、tokenize、delta action、pad state/action 等操作。
- src/openpi/serving/websocket_policy_server.py 提供 websocket 推理服务，支持 health check 和连续动作流。

**功能上怎么理解**
- 模型层做的是“看图、看状态、看语言，输出动作”。
- policies 层做的是“把不同平台的数据格式翻译成模型能吃的格式，再把模型输出翻译回机器人格式”。
- training 层做的是“定义数据集、归一化统计、权重加载、冻结策略、优化器和训练超参”。
- serving 层做的是“把训练好的 policy 变成一个能被机器人 runtime 调用的服务”。

**操作方法**
1. 安装和环境准备以 README 为准，核心是 uv sync 和可编辑安装。
2. 选一个训练配置名，例如 pi05_droid 或 pi05_libero。
3. 训练时先算 norm stats，再跑训练脚本。
4. 推理时用训练配置名和 checkpoint 目录创建 policy，然后可本地调用或起 websocket server。

最关键的使用入口都已经在 README.md 和 scripts/serve_policy.py 里串好了。比如 serve_policy 的默认路径就是读取训练配置、自动选择默认 checkpoint，然后起一个 websocket server。

如果你要，我下一步可以继续做两种更细的整理之一：
1. 按“数据流”把 pi0 / pi0-fast / pi0.5 的输入输出链路逐步画出来。
2. 按“可运行方式”把训练、离线推理、远程推理、服务端接口整理成一份可执行指南。

已创建 3 个待办事项



