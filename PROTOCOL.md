# Reproduction protocol

## Scope and source

目标论文为 Qian & Chou (2021), “A Koopman-operator-theoretical approach for anomaly recognition and detection of multi-variate EEG system,” *Biomedical Signal Processing and Control*, 69, 102911, DOI: [10.1016/j.bspc.2021.102911](https://doi.org/10.1016/j.bspc.2021.102911). 出版社摘要确认其提出 Koopman/Hankel 动力学表示，并以 Lorenz 仿真和癫痫 EEG 问题展示方法。[ScienceDirect 论文页](https://www.sciencedirect.com/science/article/abs/pii/S1746809421005085)

下文严格区分论文明确说明、Bonn 必需改动和论文材料未能唯一确定的选择。凡不能从论文明确确认的具体计算细节，不标成作者原始参数。配置中的候选值是本项目预先登记的实现选择，不会根据测试集结果调整。

## PAPER_FAITHFUL

- 论文的核心思路是用时间延迟/Hankel 表示和 SVD 获得低维 intrinsic representation，再以 DMD 近似 Koopman 线性动力学，并在 intrinsic space 中分析/重构轨迹。
- 论文提出从重构动力学相对于 attractor 的偏离构造 dynamical measurement，并用于识别动力学变化；特征与分类是分析下游，不把 Koopman eigenvalues 直接当作本项目主要分类输入。
- 论文包含 Lorenz 系统仿真和癫痫 EEG 应用。L1 参数为 `sigma=10, rho=28, beta=8/3`；异常动力学 L2 保持 `sigma=10, rho=28`，并将 `beta` 从 `8/3` 改为 `4/3`。L2 的变化是 beta 参数，不是 rho 参数。
- 论文 Lorenz 仿真采样率为 100 Hz；配置中的 0.01 s sample interval 是其倒数。
- 正文说明 rank 15–20 truncation can be sufficient，并在后续保留 “top 15 coordinates with the largest eigenvalues”。本项目的 paper-like 默认 rank 为 15；“largest eigenvalues”如何数值对应到待选 coordinates 尚未完全确定，见 `PAPER_UNSPECIFIED`。
- 正文指出较大的 time-delay embedding dimension 通常使 intrinsic dynamics 更接近线性，并比较 low/high embedding dimension；当前掌握的正文没有给出固定 `delay_dim=64`。
- 论文的 EEG 实验使用 CHB-MIT 长时程 EEG，窗口约 50 s；这与 Bonn 单段约 23.6 s 的结构不同，不能将该窗口直接套用到 Bonn。
- 本项目保留并显式实现数学步骤：delay Hankel matrix → SVD → 截断坐标 → 相邻快照矩阵 → 有限维线性算子 → eigendecomposition → intrinsic trajectory reconstruction → deviation series。

## BONN_ADAPTATION

- 数据根目录由 `config.json` 的 `bonn_root` 指定为 `E:\Python\DATA\Bonn`。用户已确认数据结构为 `A_Z/`、`B_O/`、`C_N/`、`D_F/`、`E_S/`，每类 100 段；本项目只读取 `D_F/Fxxx` 与 `E_S/Sxxx`。
- 标签固定为：`D_F/Fxxx` → label `D`、state `interictal`；`E_S/Sxxx` → label `E`、state `ictal`。不再自动判断命名族或猜测标签。当前选择总计 `N=200`，其中 100 D + 100 E。
- 一个完整 Bonn segment（预期约 4097 samples、约 23.6 s）作为一个 dynamical sample。不能把论文 CHB-MIT 的长时窗照搬到短 Bonn segment。
- 直接分析 Bonn 磁盘上已有的单通道信号；第一阶段不拆成 delta/theta/alpha/beta/gamma 频带。
- 进行 Stratified 5-fold CV，固定 seed；标准化器只在各 training fold 拟合。评估是 segment-level validation，只评价这 200 个 segment 上的 D/E discrimination。当前确认的文件结构没有可靠 patient identity 字段；不得据此声称 cross-subject、cross-patient、LOSO 或 patient-independent generalization。若未来切窗，同一源 segment 的窗口必须始终属于同一个 fold。
- 输出 Bonn 数据审计、预先固定抽样的示例图、delay-dimension 消融、特征统计和 eigenvalue 分布。六个 Koopman-derived 特征先进入 Logistic Regression、SVM 和 RUSBoost；另建 raw EEG statistics baseline。
- 下列内容是本项目针对 Bonn 的实验设计，不宣称是论文原始设置：完整 segment 样本单位、D/E 二类任务、六个总结特征、当前 CV 设计及 raw-statistics baseline。

## PAPER_UNSPECIFIED

以下参数或数值实现目前不能由论文正文唯一确认。它们在 `config.json` 中显式登记；不得依据测试集表现选择。

| 项目 | 初始实现选择 | 选择理由 |
|---|---|---|
| Hankel delay dimension | `64` 是本项目的 implementation choice；Lorenz 与 Bonn 预先指定候选 `16, 32, 64, 128` | 正文支持“较大 embedding dimension 通常更线性”及 low/high 比较，未确认固定 64 |
| Rank selection criterion 的数值实现 | paper-like rank 默认 `15`；rank 数值来源本身有论文依据，但 “top 15 coordinates with the largest eigenvalues” 如何映射到 coordinate 及如何排序仍未确定 | 不得把原文措辞改写为“top 15 singular values”；具体 criterion implementation 标为 `PAPER_UNSPECIFIED` |
| Intrinsic coordinate source | 首个 implementation candidate：`right_singular_vectors`，即 `Vh[:r, :]` 的时间序列 | 当前正文不能证明 `y=V`、`y=ΣV*` 或 `y=U` |
| DMD fitting and reconstruction convention | `A = Y2 @ pinv(Y1)`；用 eigendecomposition 演化初始 intrinsic state | 明确、可审计的有限维 DMD 实现；不是对作者代码的断言 |
| Eq. (8) numerical handling | Phase 1/2 数学模块保留 `paper_literal_finite_p` 与 `chebyshev`；Lorenz transition diagnostic 固定比较 p=20 与 abs-eigenvalue 权重 | 复数权重如何实数化、`p→∞` 的数值实现及与 Mahalanobis + Chebyshev 的关系未确定 |
| Eigenvalue weight handling | `abs(lambda_i)` 仅是实数非负权重的 implementation choice | 不属于论文定义；标记 `PAPER_UNSPECIFIED_NUMERICAL_HANDLING` |
| Attractor/reference construction | `origin` 是 paper-motivated candidate；`trajectory_centroid` 与 `training_reference_centroid` 仅作后续 sensitivity candidates | 正文说线性化较好时 trajectory tends to orbit around the origin，但 Eq. (8) 中 x/y 与 reference trajectory 如何构造仍未给出 |
| Lorenz burn-in / analysis duration | L1 burn-in 20 s；L1/L2/L1 各 50 s | 初值及时间长度为 implementation choices；不声称是论文精确 simulation duration |
| Bonn sampling rate | `173.61 Hz` | 数据说明中的近似值；按配置记录，数据审计时同时报告该值及实际信号长度 |
| Summary feature definitions | `D_power`/`raw_power` 为均方；entropy 为固定 64-bin histogram Shannon entropy（nats） | 让“power”和“entropy”有可重复的操作定义；论文/本任务未唯一规定此离散估计法 |
| Classifier details | 5-fold, seed 2021；SVM RBF；固定 Logistic Regression / RUSBoost 设定见 config | 可复现且不依测试表现搜索的初始设定；不声称是论文完整超参数 |

### Eq. (8) audit and distance definitions

论文说明 `lambda_i` 是与第 i 个维度对应的 eigenvalue。Eq. (8) 可概括为 `lim_{p→∞} (Σ_i lambda_i |x_i-y_i|^p)^(1/p)`。仍不确定的是：若 eigenvalue 为复数，作者实际怎样将其变为实数距离权重；`p→∞` 在数值代码中如何实现；以及正文提及的 Mahalanobis + Chebyshev 与 Eq. (8) 之间的精确关系。因此本项目不声称已确定论文“真实 D”。

Phase 1 已将两个候选分别实现为独立数学函数。Lorenz transition diagnostic 固定比较两者；该诊断设置不宣称是论文 Eq. (8) 唯一的数值实现：

1. `paper_literal_finite_p`: `D_p(x,y)=(Σ_i w_i |x_i-y_i|^p)^(1/p)`，`p` 可配置。若试用 `w_i=abs(lambda_i)`，必须标作 `IMPLEMENTATION_CHOICE / PAPER_UNSPECIFIED_NUMERICAL_HANDLING`，不能写成论文定义。
2. `chebyshev`: `D_inf(x,y)=max_i |x_i-y_i|`，作为无权 `p→∞` 对照。

当前不选定六项特征的主距离定义。进入实验阶段前须登记选择和理由；对另一候选做透明敏感性分析，不能按测试集指标择优。

### Intrinsic coordinates audit

对 `H=U Σ V*`：

| 量 | 含义 | 与 intrinsic coordinates 的关系 |
|---|---|---|
| `U` | 左奇异向量，位于 delay-coordinate 行/嵌入空间 | 论文当前正文不能证明 `y=U` |
| `Σ` | 奇异值构成的对角矩阵，表示奇异方向的尺度 | 不是时间轨迹本身 |
| `V` / `V*` | 右奇异向量；时间样本沿 Hankel 列排列 | `V` 的时间坐标是首个 candidate，但不能据此认定为论文定义 |
| `ΣV*` | 右奇异时间坐标再乘奇异值 | 与 `V` 不同，论文是否以此为 y 未确认 |
| DMD eigenvectors | 有限矩阵 `A` 的特征向量 | 用于表示 A 的线性演化方向，不自动等于观测轨迹坐标 |
| DMD / Koopman modes | 将模态系数映射回所选观测/状态表示的模态向量 | 与 SVD 的 U/V 和 A 的 eigenvectors 概念不同，不能互换 |
| Intrinsic coordinates `y_k` | 论文中满足 `y_{k+1}=Ay_k` 的时间变化坐标 | 具体由上列哪一个量构造仍为 `PAPER_UNSPECIFIED` |

本项目首个 implementation candidate 是 `V` 的时间序列，即 `Vh[:r, :]`；论文当前正文不足以证明 `y=V`、`y=ΣV*` 或 `y=U`。保留 `coordinate_source` 配置，且不得将任何候选称作作者已确认的定义。

### Reference attractor audit

论文指出，当 intrinsic-space dynamics 接近线性时，trajectory tends to orbit around the origin。因此 `reference_mode=origin` 是 paper-motivated candidate，不是完全任意选择。但正文仍未明确 Eq. (8) 的 `x`、`y` 分别对应什么，也未给出 reference trajectory 的数值构造，故总体标记为 `PAPER_MOTIVATED_BUT_IMPLEMENTATION_UNSPECIFIED`。`trajectory_centroid` 与 `training_reference_centroid` 只用于后续 sensitivity analysis；若用训练参考质心，只能由 training fold 数据估计并应用于验证 fold，不能根据测试集结果选择。

## Phase 1 mathematical implementation record

Phase 1 只实现和单元测试基础数学计算，不运行 Lorenz、不读取 Bonn、不生成正式实验图。

- Hankel embedding：`H[i,j] = x[i+j]`，shape `[delay_dim, T-delay_dim+1]`。
- Truncated SVD：`U, S, Vh = np.linalg.svd(H, full_matrices=False)`，截断由 `svd_rank` 控制；函数同时返回未截断 singular values 供诊断。
- 两个时间坐标候选由 `build_intrinsic_coordinates` 明确实现：`source="V"` 返回 `V.T`（其中 `V=Vh_r.T`），`source="SigmaV"` 返回 `diag(S_r) @ Vh_r`。默认 `V` 只是 `PAPER_UNSPECIFIED` 的 implementation candidate；不支持把 `U` 直接当时间轨迹。
- `svd_rank` 与 `koopman_coordinate_count` 在配置中独立记录，虽然 paper-like candidate 当前均为 15；原文 rank 15–20 truncation 与“largest eigenvalues 对应的 top 15 coordinates”不能自动视为等价。Eigenvalue magnitude 排序仅给诊断索引，不改变或筛选 `Y`。
- 对输入 `Y`（shape `[r,N]`），快照矩阵为 `Y1=Y[:,:-1]`、`Y2=Y[:,1:]`，拟合 `A=Y2 @ pinv(Y1)`，线性误差为 `||Y2-A Y1||_F / ||Y2||_F`。拟合不排序、删除、裁剪或稳定化 eigenvalues，也不使用 fbDMD/TLS-DMD。
- 求解 `eigvals, eigvecs = np.linalg.eig(A)`。离散谱保留 magnitude 和 phase；已知 `dt` 时以 `mu=log(lambda)/dt` 派生频率和 growth rate。零 eigenvalue 保留，其连续时间派生值标为 NaN/invalid，不产生可继续使用的 Inf。
- Intrinsic trajectory reconstruction 按 `y_0=y0`、`y_(k+1)=A y_k` 递推，并包含初始状态；`one_step_prediction` 返回 `A @ Y[:,:-1]`。不实现 decoder 回原始 EEG。
- deviation 候选分别实现有限 p 加权距离、Chebyshev 距离以及按时间计算的 trajectory deviation。有限 p 使用缩放/log-domain 计算避免高 p overflow；输入权重须实数、有限且非负。`make_eigenvalue_weights(..., mode="abs")` 返回 `abs(lambda)`，这是 `IMPLEMENTATION_CHOICE / PAPER_UNSPECIFIED_NUMERICAL_HANDLING`，不是论文已确认的处理。
- `trajectory_deviation` 支持 `origin`、`trajectory_centroid` 与显式 reference。`origin` 是 `PAPER_MOTIVATED_BUT_IMPLEMENTATION_UNSPECIFIED` candidate；训练集 reference centroid 需要 fold-level data flow，Phase 1 不实现。

## Phase 2 Lorenz reproduction protocol

Phase 2 是论文复现与校准，不是方法可行性筛选。与 Fig. 2–Fig. 4 的结果不一致时记录 `DEBUG`，按 Hankel orientation、intrinsic-coordinate 定义、SVD 截断、`A=Y2 @ pinv(Y1)` 方向、eigenvalue/coordinate selection、Eq. (8) 和 delay dimension 逐项检查；不因现象不明显停止后续复现。只有明确且无法解决的数学矛盾才停止。

### PAPER_FAITHFUL

- Lorenz 方程为 `dx/dt=sigma(y-x)`、`dy/dt=x(rho-z)-y`、`dz/dt=xy-beta*z`。
- L1: `(sigma,rho,beta)=(10,28,8/3)`；L2: `(10,28,4/3)`。系统切换只改变 beta。
- 采样率为 100 Hz，即 `dt=0.01 s`。
- Koopman/Hankel sanity experiment 只使用 `x(t)` observation；真实 y、z 只用于原系统轨迹展示和 ground truth。
- 与论文 Fig. 2、Fig. 3、Fig. 4 做定性对照；不以手工旋转、缩放或挑选坐标追求图形相似。

### IMPLEMENTATION_CHOICE / PAPER_UNSPECIFIED

- 初始状态 `[1,1,1]`、L1 burn-in 20 s、分析段 L1/L2/L1 各 50 s（每段 5000 samples）是固定实现设置，不声称是论文精确初值或 simulation duration。
- Lorenz 与 Bonn embedding dimensions 固定候选 `16,32,64,128`，`svd_rank=15` 始终固定。候选值是 `PAPER_UNSPECIFIED` implementation choices，不按误差结果改选，不使用 `effective_rank=min(rank,delay_dim)`。
- `svd_rank` 与 `koopman_coordinate_count` 分开记录。rank truncation 15–20 与保留最大 eigenvalues 对应的 top 15 coordinates 不自动视为数值等价；Phase 2 不以 eigenvalue 排序自动筛除或重排 `Y`。
- L1→L2→L1 连续演化：burn-in 后的 L1 末状态作为 L2 初态，L2 末状态作为 final L1 初态。阶段边界状态复用，不分别生成独立轨迹再拼接。
- Hankel column `j` 的时间锚点取 delay window 中心：`t_j=t0+(j+(delay_dim-1)/2)*dt`，其中 `t0` 为原始 observation 首样本时间。Transition boundary 在 intrinsic/deviation 图上使用同一映射解释；不以 `linspace` 假造对齐。
- `coordinate_source=V` 是首个 `PAPER_UNSPECIFIED` implementation candidate；`SigmaV` 仅作几何、linearity error 与数值 scale 的诊断对照。若某候选的量级接近 Fig. 4，只能称为 “consistent with the figure scale”，不能据此断言作者使用该坐标。
- Transition diagnostic 对完整连续 `x(t)` 固定使用 `delay_dim=64`、`svd_rank=15`、source `V` 和 origin reference candidate。比较 Chebyshev 与 finite-p (`p=20`, `abs(lambda)`)；abs 权重仍是 `IMPLEMENTATION_CHOICE / PAPER_UNSPECIFIED_NUMERICAL_HANDLING`。
- Finite-p 诊断当前将 `np.linalg.eig(A)` 返回的 eigenvalue magnitude 按返回位置依次用于 `Y` 的 coordinate 行权重。该位置对应关系只是透明的 implementation choice，正文尚不能确认其与原文第 i 个 eigen-dimension 的精确对应；该诊断不参与坐标筛选，也不被称为论文的唯一 Eq. (8) 实现。
- 记录 one-step prediction error 和固定 100-sample (1 s) short-horizon free-run error；不以 150 s 全程 free-run 误差作为唯一正确性指标。
- 若与论文预期不符，记录 `DEBUG` 和排查结果，继续复现；不得修改算法以追求明显 transition 或漂亮图。

### Phase 2 comparison outputs

- 在 burn-in 后的稳定 L1 segment 上比较四个固定 delay dimensions，`svd_rank` 始终为 15；保存 Hankel shape、全部 singular values、condition diagnostics、linearity error 和 `spectral_radius(A)`。
- 对 `V` 与 `SigmaV` 记录每个 delay dimension 的 linearity error、coordinate RMS 与 max absolute scale。
- 另保存 `Figure_Lorenz_L1_L2_L1_3D.png`、`Figure_Lorenz_embedding_comparison.png` 和 `Figure_Lorenz_linearity_error.png` 作为按项目原始图名约定的同一结果副本；图中使用预先固定的低/高 embedding dimensions，不旋转或缩放轨迹。
- 对 full L1→L2→L1 trajectory，按 center-anchored `D(t)` samples 汇总 L1 first、L2、L1 final 的 mean/median/std/max。Lorenz 是 deterministic comparison，不在本阶段做 formal hypothesis test。

## Phase 3 Bonn D/E smoke protocol

本阶段是 Bonn segment-level 数值 smoke test，只对固定 seed 抽取的 5 个 D + 5 个 E 执行动力学管线。不运行完整 200 段，不训练分类器，不做交叉验证、不做频带拆分，也不修改 Phase 1 核心数学模块。

### BONN_ADAPTATION — 固定数据映射和样本单位

- `D_F/F001...F100` 映射为 label `D` / interictal；`E_S/S001...S100` 映射为 label `E` / ictal。标签固定配置，不推断或改写。
- 文件扫描接受 `.txt` 和 `.TXT` 扩展名（大小写不敏感），仅扫描这两个目录及上述文件名前缀。
- 数据审计读取并汇总所有发现的 D/E 原始段：数量、长度分布、NaN/Inf 计数、有限幅值的最小/最大、按采样点汇总的均值/标准差。该审计不对全部 200 段运行 Hankel/Koopman 管线。
- 每类 5 段通过固定 seed `2021` 从排序后的文件列表随机无放回抽取；不基于数值稳定性或图像外观筛选。完整 segment 是一个样本；不切窗、不做 band decomposition。
- Bonn 采样率使用数据配置记录的约 `173.61 Hz`，单段预期 `4097` samples（约 23.6 s）。一个 segment 一个 dynamical sample 是数据结构适配，不是论文 CHB-MIT 窗口设置。

### Phase 3 固定管线和数值检查

- 固定 `delay_dim=64`、`svd_rank=15`、`koopman_coordinate_count=15`、`coordinate_source=V`、`reference_mode=origin`。这些是本阶段预先固定的 smoke 设置；`delay_dim=64` 与 `V` 的论文对应关系仍按配置 provenance 记录。
- 对每个抽中段执行 `x → Hankel → truncated SVD → Y → A=Y2 @ pinv(Y1) → eigenvalues → D(t)`。分别计算 primary Chebyshev 和次要 finite-p (`p=20`, `abs(lambda)` weights) 候选。`abs(lambda)` 的实数权重转换及其位置与坐标的对应仍是 implementation choice，不宣称为论文唯一 Eq. (8) 实现。
- 每个 D 序列和 raw segment 都提取 max/min/mean/population std/mean-square power/64-bin natural-log histogram entropy。直方图固定 bin 数为 64、范围取该条信号自己的观测范围；这是可复现的 implementation choice，不能写成论文指定的 entropy estimator。
- 逐样本记录 Hankel shape、全部 singular values、linearity error、spectral radius、NaN/Inf 标志、eigendecomposition 状态与错误原因。某样本失败时保留失败行并继续处理其他样本；不得静默跳过。
- Hankel 列时间使用 delay window center：`t_j=(j+(delay_dim-1)/2)/fs`。
- smoke 的“通过”表示数值管线和 D(t)/特征可运行，不等价于 D/E 的正式统计显著性或分类能力。若数值稳定而 separation 不清楚，状态记为 `SMOKE PASS, SEPARATION UNCERTAIN`，不得因此重写算法。

### Phase 3 outputs

- `outputs/bonn_smoke/data_audit.json`：D/E 原始数据审计。
- `sample_manifest.csv`：固定随机种子抽取的 10 段 ID、标签和相对路径。
- `per_sample_metrics.csv`、`features.csv`、`config_snapshot.json`、`smoke_summary.json`。
- `Figure_BonnSmoke_raw_examples.png`、两种 D(t) 样本网格、intrinsic trajectory 网格和 feature table。

## Pre-registered run sequence

1. Phase 0: 建立协议、配置和项目说明；不运行实验。
2. Phase 1: 核心数学模块与单元测试（本阶段）；只有全部单元测试通过后才标记 complete。
3. Phase 2: 按论文参数复现 Lorenz，并对 Fig. 2–Fig. 4 校准；出现差异时进入 DEBUG 排查，不因缺少预期分离而停止。
4. Phase 3: Bonn 5 D + 5 E smoke test。记录 PASS、`SMOKE PASS, SEPARATION UNCERTAIN` 或 FAILED；完成后等待显式授权。
5. Phase 4: 固定配置处理完整 D/E 数据，进行描述统计、效应量、Holm 校正、raw baseline 对照和简单分类；根据证据记录 `SUPPORTIVE`、`REPRESENTATION_ONLY` 或 `NEGATIVE`，不设置准确率门槛。
6. Phase 4 完成后停止。任何其他数据集或后续实验须另行明确授权。

输出目录中的运行结果不提交 Git。每个获准阶段单独打包该阶段代码、文档和产物。

## Phase 4 Bonn D/E full validation protocol

本阶段已获准并完成。实验单位为单个完整 Bonn segment，共 `100 D/interictal + 100 E/ictal = 200`。数据中没有可靠可用的 patient identity，因此结果只能称为 **segment-level validation**；不得据此声称 cross-patient、cross-subject、LOSO 或 patient-independent generalization。

### BONN_ADAPTATION — 固定数据和动力学管线

- 只扫描 `D_F/F001...F100` 和 `E_S/S001...S100`，大小写不敏感地接受 `.txt/.TXT`。label 与 state 固定为 D/interictal 和 E/ictal；不自动推断标签。
- 一个完整的 4097-sample 段作为一个 dynamical sample；以存储的单通道信号直接处理，采样率 `173.61 Hz`。不切窗、不做频带拆分、不做逐段 z-score。
- 与 Phase 3 完全相同的预注册数值设置：`delay_dim=64`、`svd_rank=15`、`koopman_coordinate_count=15`、`coordinate_source=V`、`reference_mode=origin`。Chebyshev 是 primary deviation；secondary 是 `p=20` 的 finite-p candidate，权重采用 `abs(lambda)`。这些均未在本阶段根据结果修改。
- 每段依次执行 Hankel、截断 SVD、V-based intrinsic coordinates、`A=Y2 @ pinv(Y1)`、eigendecomposition、两类 `D(t)` 和 6 个 summary features。raw baseline 为原始段上的 mean、population std、mean-square power 和固定 64-bin natural-log histogram entropy。直方图范围使用当前序列观测范围；这是实现选择。
- 每样本保留 Hankel shape、奇异值、linearity error、spectral radius、NaN/Inf、SVD/eig 状态和失败原因。仅当所有 200 段成功时才开展统计和分类，避免静默丢样本。

### 统计分析和 multiple-comparison family

- 六个 primary Chebyshev 特征分别报告 D/E 的 n、mean、sample SD、median、IQR、min/max；使用双侧 Mann–Whitney U 检验，效应量为 Cliff's delta `E minus D`。六个 Koopman 检验作为一个 family 做 Holm 校正。
- 四个 raw baseline 特征使用同一描述统计、检验和效应量；四个 raw 检验独立构成一个 Holm family。保留 raw p 与 adjusted p，不按显著性删特征。
- `linearity_error` 与 `spectral_radius` 分别按 D/E 报告 mean、sample SD、median、IQR、range；它们属于数值/动力学诊断。
- 固定 scale control 对 Phase 3 manifest 中预先固定的 F094 和 S026 比较 `x(t)` 与 `10*x(t)`。它使用同一 pipeline，不按标签或效果挑样本，不改算法。

### 固定分类协议

- 预注册三组特征：六个 Chebyshev Koopman features、四个 raw features、二者合并的十个 features。不得特征筛选。
- 分类器限定 LogisticRegression (`C=1`, lbfgs, max_iter=2000)、RBF SVM (`C=1`, `gamma=scale`, probability enabled) 和 RUSBoost (DecisionTree stump; 50 estimators; learning rate 1; SAMME.R; fixed seed 2021)。参数和包版本在 `config_snapshot.json` 保存。
- 使用 `RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2021)`；所有 9 种模型/特征组合共用同一组 25 folds。LR 和 SVM 的 StandardScaler 在各 training fold 内拟合；RUSBoost 不缩放。
- 每 fold 保存 accuracy、balanced accuracy、sensitivity、specificity、F1、ROC-AUC，并保存 split manifest 与 OOF 预测。汇总为 25-fold mean ± sample SD。相同数据在重复 CV 中反复出现，fold 差异仅用于描述 split 稳定性，不当作 25 个独立外部队列或独立推断样本。
- 结果判读不设准确率通过阈值：`SUPPORTIVE` 表示 Koopman 分离与合并表示的额外判别信息均得到本段级实验支持；`REPRESENTATION_ONLY` 表示 Koopman 可表征 D/E 但未显示相对 raw baseline 的稳定补充；`NEGATIVE` 表示 smoke separation 未在全量稳定出现。三类都是有效结果，均不授权增加复杂模型或转向其他数据集。

### Phase 4 output index

- `outputs/bonn_full/config_snapshot.json`, `features.csv`, `per_sample_metrics.csv`, `numerical_audit.json`, `linearity_spectral_summary.csv`
- `statistical_tests.csv`, `raw_statistical_tests.csv`, `scale_invariance_control.json`
- `cv_split_manifest.csv`, `cv_metrics.csv`, `oof_predictions.csv`, `classifier_summary.csv`, `classifier_pairwise_comparison.csv`
- 所需图像 `Figure_BonnFull_*.png` 和 `RESULTS_PHASE4.md`
