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
