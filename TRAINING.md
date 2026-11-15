# B 路径结构细化 v1（2026-11-15，代码接入）

本轮实现基于 `master`，没有合并或更新 `main`。方案来源为 `idea/B路径合成改造方案_无标注结构细化_v1.md`。本机只有 PyTorch 2.11.0 CPU，仓库未提供可核实的原模型权重及 Kaggle 训练数据，因此**尚未启动正式预热或 20k 联合训练，也没有生成质量或真实分割提升的实验结论**。下文历史实验记录不代表此次新模型的训练结果。

## 已实现内容

- 保留原 A/C 及 Gε/Gs 共享参数，B 的初始输出 I0 后接同一个 R，默认迭代 3 步。R 为基础通道 32、两次下采样的 U-Net，共 **484,897 个参数**，8 通道输入，多尺度注入目标几何，零初始化输出头，使用 alpha=0.1 的 logit 残差更新。
- 条件全部来自本批二值分形 F。Gs 仍接收原 2 通道 one-hot；R 接收占据区域、边界、中心线、内正外负的截断距离图。距离半径在 384 分辨率为 16 像素。几何在随机增强后重新计算，空掩膜、全前景、细支有有限输出。
- C 接收最终细化图。新增最终 soft Dice、软边界 Dice；联合阶段随机抽取一个非最终细化步骤执行同一 C，中间几何系数在 2000 步内从 0 渐增至最终项的 0.25。首版不引入 soft-clDice、教师或真实血管标注。
- 整体 LSGAN 系数 0.2；局部 PatchGAN 为基础通道 32、2 层，系数 0.05。每张图抽取两个 128×128 块，99% 以上像素位于参考图有效视野，真实与合成使用相同坐标。没有可用块时明确报错，须统一降低 patch_size。
- 前 2000 次更新仅训练 R 和两个 D，原 Gε/Gs/H 冻结；原模块使用 eval 模式，避免 dropout 和 spectral normalization 的缓冲状态漂移。随后联合训练至 20000 步。梯度始终可以经过 C 回到 R。
- 原 NCE 的 detach 和 sum 归约保持原样，其直接训练 H，不作为几何监督。H 的输入单独固定为 256，避免 384 分辨率改变原 PatchSampleF 的 MLP 形状。原始/加权损失均记入 `refinement_metrics.jsonl`；NCE_raw 表示应用原 lambda_NCE 后的基线项，其余 raw 项为未加系数的均值损失。
- 新实验关闭逐位置 VGG 和旧 appearance 项；沿用 quality 配置的采样权重（等级 0 为 1.0，其余默认 0.3）。分形独立随机抽取，不读取旧 conditions。

## 数据与兼容性

新配置为 `config/train_kaggle_refine_v1.json`，默认 384×384、batch=1、单设备。原 256 的 attention 分辨率 16 映射到 384 的 24，保留相同模块层级和参数键。若统一降到 256 做接入实验，需要同时设置 dataset/model 的 image_size=256、attn_res=[16]，不能将分辨率收益计为 R 的收益。

程序读取现有 `split_manifest.json` 的 train 列表，再按受试者分组留出约 16 张内部无标注监测图；识别 `_left/_right/_OS/_OD/_L/_R` 等文件名后缀，同组双眼一并留出。若文件名无法表达真实受试者关系，启动前应核实清单。监测图不参与优化，固定分形来自程序生成的 trainA。清单、输入 SHA256 和训练/监测受试者列表保存在 `monitor_manifest.json`，运行配置保存为 `config_snapshot.json`。

参考图保留完整有效眼底，裁黑边后正方形填充；独立目标掩膜按完整坐标域最近邻缩放，不按血管包围盒放大。训练/内部监测将目标与参考有效视野求交，并记录处理前后像素数；用户测试默认不裁掉目标细支。对应参考模式对掩膜共用原图变换。裁剪、填充、像素中心变换矩阵及训练增强矩阵均记录。

`num_workers=0` 是此首版恢复协议的显式约束：完整保存 epoch 位置、epoch 开始和当前的 Python/NumPy/Torch/CUDA 随机状态，恢复时重建采样顺序，跳过已完成批次后恢复模型随机流。监测使用独立噪声生成器并还原训练随机状态。此约束避免多进程预取状态遗漏；跨设备/不同 CUDA 算子版本仍应重新做复现检查。

## 启动与恢复

先在训练主机安装与 GPU 相符的 PyTorch，再安装新增依赖：

```bash
python -m pip install -r requirements-refinement.txt
```

核实配置中的数据和质量 CSV 路径。迁移需要实际存在的、原 2 条件通道同架构的 `<prefix>_G.pth` 与 `<prefix>_H.pth`。程序记录权重 SHA256 及全部缺失/多余/形状不匹配键；只允许新增 refiner 键缺失，其余不兼容立即失败。默认不允许从随机原网络直接开始新模块预热，不填写猜测的历史权重前缀。

```bash
PRETRAINED=/verified/original/checkpoint/Ixxxxx_Ex GPU=0 bash run_train_refine_v1.sh
# 等价入口
python main.py --config config/train_kaggle_refine_v1.json --phase train --gpu_ids 0 \
  --pretrained /verified/original/checkpoint/Ixxxxx_Ex

# 同架构新实验完整恢复：不再传 pretrained
RESUME=/new/experiment/checkpoint/I1000_E1 GPU=0 bash run_train_refine_v1.sh
```

每 1000 步、预热边界及退出时保存 `<prefix>_refine.pth`（G 内含 R、H、整体/局部 D、四个优化器、计数、阶段、采样位置、随机状态、配置签名、迁移报告），另导出 `<prefix>_G.pth` 用于推理。完整状态以临时文件加原子重命名写入；至少保留最近 3 组，另保留预热边界、每 5000 步里程碑和最终上限。旧六文件优化器状态不能用于新实验 resume；原配置继续沿用旧格式。

首次更新及每 1000 步输出内部固定监测结果；不会读取 `test_input`，不会启动论文中的外部教师。每步记录时间、原始/加权损失、R 梯度范数、阶段和 CUDA 峰值显存。正式训练前须在目标 GPU 实测 384 的显存和吞吐；本机 CPU 接入耗时不能作为 GPU 训练预算。

## 训练后测试

```bash
python test_refinement.py --config config/train_kaggle_refine_v1.json \
  --weight /new/checkpoint/I20000_Ex --manifest /inputs/cases.json \
  --baseline_weight /verified/original/checkpoint/Ixxxxx_Ex \
  --output_dir /new/results/refinement_comparison --seed 2026 \
  --timesteps 200 800 1600 --refine_steps 3
```

manifest 支持以下形式，路径相对清单文件解析：

```json
{"cases": [
  {"id": "independent_01", "reference": "reference_a.png", "mask": "target_b.png", "mode": "independent"},
  {"id": "corresponding_01", "reference": "reference_b.png", "mask": "target_b.png", "mode": "corresponding"}
]}
```

也支持 `--reference_dir ... --mask_dir ... --mode independent`，按明确相同文件 stem 配对；任意独立组合使用 manifest。掩膜支持单通道 0/1 或 0/255，正值为血管，`--threshold` 默认 0.5，不自动猜极性。输出目录必须不存在。

输出原始参考/目标、坐标变换后的有效掩膜、四通道几何、I0…I3、中心放大对照、C 预测、固定噪声换掩膜及换参考图的诊断图，以及含输入/权重哈希、配置、时间、显存、设备、阈值的 JSON。没有直方图匹配或低频替换后处理。`--baseline_weight` 可选但建议提供：联合训练后的 I0 已被更新，不能独自充当原模型消融基线。循环 Dice 仅表示训练中 C 的内部一致性，不是独立分割准确率。

## 方案复核补充（2026-11-15）

逐批记录 `training_transforms.jsonl`，包含实际参考图/随机分形路径、参考图裁剪/填充/缩放矩阵、目标全域映射矩阵、增强矩阵、有效视野处理前后目标像素数、step/epoch/批次位置。测试验证中途恢复后的这些记录与连续运行一致。

每份监测报告记录当前网络全部状态张量的 SHA256 指纹，独立于是否已经导出检查点文件；测试入口另保留权重文件 SHA256。敏感性对照优先使用清单中的另一张真实参考图或另一张目标掩膜，保持本例噪声不变。也可在案例中指定 `alternate_reference`、`alternate_mask`（路径相对清单）；缺少其他输入时使用翻转作为回退，报告明确区分回退类型，并导出实际替代输入及来源/哈希/变换。监测结束后逐模块恢复 train/eval 模式，避免改变预热阶段原模块的冻结状态。

## 本次实际验证

运行 `python -m pytest tests/test_refinement_pipeline.py -q`，**14 项测试通过**，覆盖几何边界条件、旧 B 一致性、零初始化、各步骤几何梯度、局部块对称采样、坐标/增强/受试者划分、预热与联合更新、严格迁移、完整保存恢复、监测随机流、真实 U-Net/SPADE 前向及反向传播。训练入口另以小网络和临时无标注数据执行中途打断/恢复，比较连续运行的 G/H/整体 D/局部 D 全部参数。

这些是接入正确性检查；真实 384 长训、目标 GPU 显存、旧权重实物迁移以及独立真实标注评估尚待训练环境具备后执行。本轮不据此声称图像更真实或真实分割指标更高。

---

# 眼底数据训练记录

这份说明只记录我们在官方 C-DARL 上做的眼底实验：用了哪些数据、用哪份配置启动、权重存在哪里、怎么看进度。官方论文和原始 XCAD 训练见 [README.md](README.md)。

数据不在本仓库里，在上一级 `../data`。训练不复制图像，只把真实图路径写进 manifest，分形伪标签继续用 `data/Dataset_Fundus/train/trainA`。

环境是 `/home/WVK/anaconda3/envs/medsam2`。命令都在 `MEDIA_CDARL` 目录下执行。

## 数据怎么进模型

`data/XCAD_dataset.py` 在训练集上读两份互不配对的图：

- `A`：真实眼底图。设置了 `source_dir` 时，从 `dataroot/split_manifest.json` 的 `train` 列表读绝对路径。
- `F`：`train/trainA/*.png` 里的分形血管伪标签。真实图和伪标签按索引取模配对，不是同一张图的标注。

这些实验没有血管金标准，所以不算 Dice。`main.py` 在验证集为空时跳过 Dice，继续训练。

`step_start_ema: 5000` 不是保存间隔。它只决定第 5000 步之后开始更新 EMA。按步保存必须另写 `save_checkpoint_freq`。没写这个字段时，只按 `save_checkpoint_epoch` 在整轮结束时保存。

## 已跑过的实验

| 实验 | 数据 | 启动 | 停止条件 | 结果 |
| --- | --- | --- | --- | --- |
| 眼底 UWF/FP | 当时挂进 `Dataset_Fundus` 的眼底图 | 2026-11-01 16:31，GPU 0 | 配置写 150 epoch，实际停在第 100 epoch | 有权重，未跑完 150 epoch |
| 全部 Kaggle | `../data/kaggle/extracted/train` | 2026-11-05 10:03，GPU 0 | 100000 step | 已完成 |
| Kaggle EyeQ q0 | 同上目录中的 7499 张优质图 | 2026-11-08 12:09，GPU 0 | 100000 step | 已完成 |
| 五数据集混合 | kaggle、MESSIDOR、REFUGE、FIVES、DomainB_FP | 2026-11-08 16:56，GPU 1 | 配置为 100000 step | 约 82340 step 时进程消失，没有权重 |

### 眼底 UWF/FP

配置是 [config/train_fundus.json](config/train_fundus.json)。实验目录：

`experiments/CDARL_fundus_UWF_FP_261101_163137`

最后一次常规保存是 `checkpoint/I100700_E100`。这份配置现在的 `resume_state` 指向该权重。直接再用它启动会从 epoch 100 继续，而 `n_epoch` 是 150，所以只会再跑 50 个 epoch。想从头训练时把 `resume_state` 改成 `null`。

### 全部 Kaggle

实验名 `CDARL_kaggle_gray`，数据名 `KaggleOnly`，来源是 `../data/kaggle/extracted/train`。

完成目录：

`experiments/CDARL_kaggle_gray_261105_100342`

最终权重：

`checkpoint/I100000_E4`

同名的 10:03:01 目录是同一次启动前的空目录，不用。

### Kaggle EyeQ q0

配置是 [config/train_kaggle_eyeq_q0.json](config/train_kaggle_eyeq_q0.json)。`source_dir` 仍指向全部 Kaggle 训练图，实际用哪些图由 `data/Dataset_Kaggle_EyeQ_q0/split_manifest.json` 决定，本次是 7499 张。

完成目录：

`experiments/CDARL_kaggle_eyeq_q0_261108_120921`

最终权重：

`checkpoint/I100000_E14`

这次配置写了 `save_checkpoint_freq: 5000`，所以中断后可以从最近的 5000 步权重恢复。`n_epoch: 150` 不会跑满，`max_steps: 100000` 先到。

当时的启动命令是：

```bash
/home/WVK/anaconda3/envs/medsam2/bin/python -u main.py -p train \
  -c config/train_kaggle_eyeq_q0.json -gpu 0
```

这是前台命令。要断开 SSH 后继续跑，需要套下面混合训练用的 `setsid nohup`，或者自己放到 tmux 里。

### 五数据集混合

配置是 [config/train_mixed_fundus.json](config/train_mixed_fundus.json)。启动脚本是 [scripts/start_mixed_train.sh](scripts/start_mixed_train.sh)。

脚本会先运行 [scripts/build_mixed_manifest.py](scripts/build_mixed_manifest.py)，扫描这五个目录并覆盖 `data/Dataset_Fundus/split_manifest.json`：

- `../data/kaggle`
- `../data/MESSIDOR`
- `../data/REFUGE`
- `../data/FIVES`
- `../data/DomainB_FP`

扫描会跳过目录名像 `mask`、`label`、`gt`、`segmentation` 的文件夹。2026-11-08 16:48 的清单是：

| 数据集 | 图像数 |
| --- | ---: |
| kaggle | 35126 |
| MESSIDOR | 1200 |
| REFUGE | 2000 |
| FIVES | 800 |
| DomainB_FP | 619 |
| 合计 | 39745 |

同一天 00:47 扫到的 MESSIDOR 是 3220 张，16:48 只剩 1200 张。清单数量以当次扫描结果为准，不是数据集的固定规模。分形标签是 1000 张。

启动：

```bash
GPU_ID=1 USER_LAUNCH=1 bash scripts/start_mixed_train.sh
```

`USER_LAUNCH=1` 是必须的，避免误启动。`GPU_ID` 不写时默认 0。脚本用 `setsid nohup` 把训练放进独立会话，关闭 SSH 不会发送 SIGHUP。

看进度不要看 `logs/mixed_launch/launch.log`，那里会接上旧运行的输出。去本次实验目录：

```bash
tail -n 1 experiments/CDARL_mixed_fundus_261108_165631/checkpoint/loss_log.txt
```

新一次启动会生成新的 `experiments/CDARL_mixed_fundus_<时间戳>`。

16:56 这次读到了 `max_steps: 100000`，但配置里没有 `save_checkpoint_freq`。因此它只在第 10 个 epoch 保存，第一次保存大约在 397450 step。进程在 2026-11-08 22:21 前后消失，最后日志是第 3 epoch 的 2850/39745，合计约 82340 step。没有 Python 报错，也没有 `End of training`，checkpoint 目录里没有 `.pth`。这次不能恢复，只能重跑。

重跑前应先在配置里加上：

```json
"save_checkpoint_freq": 5000
```

否则再次中断仍然不会留下权重。

## 测试合成

`test_input` 里要有成对的 `*_image.png` 和 `*_mask.png`。测试走训练时的合成路径，不是官方 `main.py -p test` 的血管 Dice。

全部 Kaggle 权重：

```bash
/home/WVK/anaconda3/envs/medsam2/bin/python -u test_synthesis.py \
  -c config/train_kaggle_eyeq_q0.json \
  --input_dir test_input \
  --output_dir test_output/kaggle_all_100k \
  --weight experiments/CDARL_kaggle_gray_261105_100342/checkpoint/I100000_E4 \
  --gpu 0
```

Kaggle EyeQ q0 权重：

```bash
/home/WVK/anaconda3/envs/medsam2/bin/python -u test_synthesis.py \
  -c config/train_kaggle_eyeq_q0.json \
  --input_dir test_input \
  --output_dir test_output/kaggle_eyeq_q0_100k \
  --weight experiments/CDARL_kaggle_eyeq_q0_261108_120921/checkpoint/I100000_E14 \
  --gpu 0
```

对比时看两个输出目录里同一编号的 `*_panel.png`。图中从左到右是原图、条件血管掩码、合成图。`cycle_dice` 用的是同一个已训练分割器，不能当成独立精度。

## 恢复训练

`path.resume_state` 写权重前缀，不带 `_G.pth`。例如：

```text
experiments/CDARL_kaggle_eyeq_q0_261108_120921/checkpoint/I100000_E14
```

训练阶段会加载 `_G.pth`、`_H.pth`、`_Da.pth` 和三个优化器状态。优化器文件里的 epoch 和 step 会成为新的起点。如果该 epoch 已经达到 `n_epoch`，或者 step 已经达到 `max_steps`，启动后会立刻结束。
