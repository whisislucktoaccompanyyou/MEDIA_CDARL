# Kaggle 灰度 C-DARL 训练

本轮仅使用 `../data/kaggle/extracted/train/*.jpeg`，从头初始化；不使用旧照片或旧权重。保留作者扩散、循环、LSGAN、MaskPatchNCE 损失与权重。512 灰度图经作者隔点抽样为 256 输入，batch=1，lr=1e-5，最多 100000 个计数 step（第一个为作者的数据依赖初始化）。患者编号分组划分 90/5/5，清单见 `data/Dataset_Kaggle/split_manifest.json`；val/test 照片暂保留，不参与训练。照片在线裁黑边、等比补边。1000 张分形标签沿用此前本地生成器，不宣称是作者原始分形标签库。

启动（关闭 SSH 后继续运行）：

```sh
./run_train_kaggle.sh
tail -f logs_launch/train_kaggle.log
```

启动脚本用 setsid/nohup 和 flock 避免重复实例；每次保留独立日志。不能抵御重启或系统终止。每 5000 step 保存 G/H/D 和优化器，训练结束保存最终权重；初次 step 2 和每 5000 step 输出 `test_input` 的 10 对图像、掩膜的固定 seed=2026、t=200/800/1600 合成图，位于本次 experiments 下的 results/step_*。

`test_input` 只用于定期效果展示，不参与梯度更新，不自动根据它选模型。循环 Dice 使用模型自己的分割器，不是独立精度。condition_change_mae 是同一参考/噪声下水平翻转条件后的响应差异，不单独证明血管遵循程度。报告与拼图应结合检查。

对比学习保留作者 detach：正/负预测特征都 detach，所以对比损失经分形查询特征训练 H，不直接向生成器反传。记录 H 梯度范数、正负特征差异、预测掩膜标准差和前景比例辅助排查退化；不因旧日志常数就改变原始损失。

单独测试某个 checkpoint（传入不带 _G.pth 的前缀）：

```sh
/home/WVK/anaconda3/envs/medsam2/bin/python test_synthesis.py \
  -c config/train_kaggle.json --weight experiments/本次目录/checkpoint/I5000_E1 \
  --input_dir test_input --output_dir test_output_kaggle --seed 2026 --timesteps 200 800 1600
```

掩膜支持 0/1、0/255；按 0.5 阈值二值化，再最近邻缩放及 one-hot。概率图需先保存为对应的单通道 8-bit 概率 PNG；多类别标签需先提取血管类别。本轮输出为灰度图，需要参考照片，不是仅凭掩膜还原原始照片。

本脚本每次启动新实验，不自动断点续训。如中断后需接续，可将 config/train_kaggle.json 的 path.resume_state 设为完整 checkpoint 前缀，恢复全部网络/优化器/计数；max_steps 是恢复后的总计数上限。保持相同数据清单；当前不保存 DataLoader/RNG 精确位置，因此不保证逐位复现。
