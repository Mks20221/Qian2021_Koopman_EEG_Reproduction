# Qian & Chou (2021) Koopman EEG reproduction

本项目独立复现并检查 Qian 与 Chou 2021 年提出的 Koopman/Hankel 动力学表示，并将核心流程适配到本地 Bonn 癫痫 EEG 数据，仅比较 interictal 与 ictal 两类。目标是检验重构动力学偏差特征是否含有区分信息，不以追求最高分类准确率或提出新模型为目标。

论文：Qian, S. & Chou, C.-A., “A Koopman-operator-theoretical approach for anomaly recognition and detection of multi-variate EEG system,” *Biomedical Signal Processing and Control*, 69 (2021), 102911. [DOI](https://doi.org/10.1016/j.bspc.2021.102911) · [出版社页面](https://www.sciencedirect.com/science/article/abs/pii/S1746809421005085)

Lorenz sanity check 按论文参数从 L1 切换到 L2 再回到 L1：L1 为 `(sigma, rho, beta)=(10, 28, 8/3)`，L2 为 `(10, 28, 4/3)`。本项目 paper-like rank 默认值为 15；`delay_dim=64` 是可配置的 implementation choice，不是论文给出的固定值。

## 数据

Bonn 数据从 `config.json` 中的 `bonn_root` 读取，默认指向 `E:\Python\DATA\Bonn`。用户已确认目录包含 `A_Z`、`B_O`、`C_N`、`D_F`、`E_S` 五个文件夹，每个 100 段。本项目固定使用 `D_F/Fxxx`（label `D`，interictal）和 `E_S/Sxxx`（label `E`，ictal），共 200 段；不自动判断或改写标签。运行 Bonn 流程时先生成数据审计。不会下载 CHB-MIT 或其他 EEG 数据集，也不会读取其他平级项目中的代码或结果。

每个完整 Bonn segment 作为一个样本。第一阶段直接使用磁盘上已有的单通道信号，不做五频带拆分。Bonn 流程是针对数据长度的适配，不称为论文参数的严格复现。验证单位是 segment；当前不据此声称 cross-subject、cross-patient、LOSO 或 patient-independent generalization。

## 环境

使用 Python 3.10 或更新版本。依赖包含 NumPy、SciPy、Matplotlib、pytest、scikit-learn 与 imbalanced-learn，安装依赖：

```powershell
python -m pip install -r requirements.txt
```

Phase 4 的 scikit-learn 和 imbalanced-learn 版本已固定在 `requirements.txt`，以复现分类器实现。

## 运行命令

Phase 1 提供显式 Hankel/SVD、intrinsic coordinate candidates、least-squares DMD、eigenvalue diagnostics、linear reconstruction 和 deviation candidate functions。单元测试命令：

```powershell
pytest -q
```

Phase 2 Lorenz 入口按固定配置生成连续 L1 → L2 → L1 trajectory，比较预先固定的 delay dimensions，并输出重构/偏差诊断及图表：

```powershell
python scripts/run_lorenz.py
```

结果保存在 `outputs/lorenz/`，包括配置快照、原始仿真轨迹、embedding 与 coordinate-source 指标、D(t) 分段汇总、重构误差 JSON 和 Figure_01–Figure_08。Phase 2 用论文图做定性/定量校准；出现差异时记录为 `DEBUG` 并排查数学实现，不把差异设为停止整个复现的 Gate。该脚本不读取 Bonn 数据。

Phase 3 只从固定的 `D_F/Fxxx` 与 `E_S/Sxxx` 中，用 seed 2021 各随机抽取 5 段做数值 smoke test。它会审计 D/E 文件集、保存抽样清单，使用 `delay_dim=64`、`rank=15`、`V` 坐标和 origin reference 产生两种 D(t) 及 summary features；不会执行全量 Koopman 处理、分类或交叉验证：

```powershell
python scripts/run_bonn_smoke.py
```

Phase 3 产物写入 `outputs/bonn_smoke/`，包括数据审计、10 个样本清单、逐样本数值诊断、特征 CSV 和 smoke figures。遇到单样本失败会记录原因并继续处理其余抽样样本。

Phase 4 对全部 D/E 200 个 segment 使用与 smoke 相同的固定动力学管线，输出六个 Koopman 特征、四个 raw baseline 特征、数值审计、Mann–Whitney U 与 Holm 检验、尺度控制及固定的 repeated stratified 5×5 segment-level CV：

```powershell
python scripts/run_bonn_full.py
```

完整结果保存在 `outputs/bonn_full/`，分析说明见 `RESULTS_PHASE4.md`。分类器仅为 LogisticRegression、SVM 和 RUSBoost；所有缩放只在训练折拟合。Bonn 数据没有可靠 patient identity，因此结果仅表示这 200 个 segment 上的 D/E discrimination，不能解释为 cross-patient generalization。Phase 4 不做调参、特征选择或频带拆分。

## 预期输出

Lorenz、Bonn、图表和表格产物保存在 `outputs/` 下。运行产物默认由 Git 忽略；交付压缩包会包含项目代码、文档和对应阶段已生成结果。
