1. 创建/root/data1/zjj/Grokking_Formulation/src/qk_circut.py，取100，500，1000，5000，10000，50000，990000七个步数的权重矩阵，计算有效的注意力交互矩阵 $A = W_E W_Q W_K^T W_E^T$（维度 $p \times p$），对这个 $p \times p$ 矩阵进行 二维离散傅里叶变换 (2D DFT)，将数据保存为/root/data1/zjj/Grokking_Formulation/data/x+y/qk_circut.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/qk_circut.py，使用qk_circut.csv文件，绘制2D频谱图，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，执行上述任务。

2.  修改/root/data1/zjj/Grokking_Formulation/src/gini_ipr.py，使用/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints中保存的权重文件，提取输入嵌入矩阵W_e，将其铺平为一维向量，DFT转换成频域，再计算其Gini系数和IPR逆参与率，将数据保存为/root/data1/zjj/Grokking_Formulation/data/x+y/gini_ip.csv，然后修改/root/data1/zjj/Grokking_Formulation/src/plot/gini_ip.py，使用gini_ip.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc以及Gini/IPR变化，横轴为训练步数，纵轴为相似度，三y轴，横轴依然采用对数刻度，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，执行上述任务。

修改/root/data1/zjj/Grokking_Formulation/src/qk_circut.py，取100，500，1000，5000，10000，50000，990000七个步数的权重矩阵，计算有效的注意力交互矩阵 $A = W_E W_Q W_K^T W_E^T$（维度 $p \times p$），对这个 $p \times p$ 矩阵进行 二维离散傅里叶变换 (2D DFT)，将数据保存为/root/data1/zjj/Grokking_Formulation/data/x+y/qk_circut.csv，然后修改/root/data1/zjj/Grokking_Formulation/src/plot/qk_circut.py，使用qk_circut.csv文件，绘制2D频谱图不变，只是改为7个点的变化，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，执行上述任务。

3. 创建/root/data1/zjj/Grokking_Formulation/src/intrinsic_dimension.py，使用/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints中保存的全部权重文件，提取输入嵌入矩阵W_e和输出嵌入W_u，使用TwoNN方法计算其内在维度（ID），以及计算有效矩阵W_QK=W_Q（W_K）T,对齐进行SVD分解计算有效秩和谱熵，将数据保存为/root/data1/zjj/Grokking_Formulation/data/x+y/intrinsic_dimension.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/intrinsic_dimension.py，使用intrinsic_dimension.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc以及内在维度变化，横轴为训练步数，纵轴为acc和维度数，双y轴，横轴依然采用对数刻度，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，执行上述任务。

基于之前的实验现象：Gini系数基本保持不变，IPR在grokking时刻陡降，psd热力图不随步数变化，余弦相似度先上升后不变再在grokking时刻上升，并且2D频域图确实在grokking时刻出现明显对角线，W_u的内在维度（ID）先升（训练acc迅速上升）后降（测试acc迅速上升），W_e的ID是随着过拟合上升，后在grokking前下降，W_e的谱熵一直上升，W_u谱熵出现双重下降现象（分别对应训练和测试acc的迅速上升），分析背后的机制，并进一步探究。

4. 创建/root/data1/zjj/Grokking_Formulation/src/phase_coherence.py，使用/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints中保存的全部权重文件，提取输入嵌入矩阵W_e对 $W_E$ 的每一列进行 DFT，得到复数频谱 $F(k) = |A|e^{i\phi}$，计算相位 $\phi(k)$ 与频率索引 $k$ 的线性相关性 $R^2$，将数据保存为/root/data1/zjj/Grokking_Formulation/data/x+y/phase_coherence.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/phase_coherence.py，使用phase_coherence.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc以及内在维度变化，横轴为训练步数，纵轴为acc和R^2，双y轴，横轴依然采用对数刻度，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，执行上述任务。

4. 创建/root/data1/zjj/Grokking_Formulation/src/fourier_basis.py，使用/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints中保存的全部权重文件，提取输入嵌入矩阵W_e和输出嵌入矩阵W_u，将权重矩阵 $W$ 投影到傅里叶基矩阵 $F$ 上，计算投影系数的稀疏度（如 L1/L2 比值），将数据保存为/root/data1/zjj/Grokking_Formulation/data/x+y/fourier_basis.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/fourier_basis.py，使用fourier_basis.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc以及稀疏度变化，横轴为训练步数，纵轴为acc和稀疏度，双y轴，横轴依然采用对数刻度，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，执行上述任务。

5. 创建/root/data1/zjj/Grokking_Formulation/src/circuit_Competition.py，使用/root/data1/zjj/Grokking_Formulation/data/x+y/checkpoints中保存的全部权重文件，提取输入嵌入矩阵W_e和输出嵌入矩阵W_u，定义两个子空间：$S_{memo}$（由随机大梯度样本定义）和 $S_{fourier}$（由 DFT 基定义），追踪模型权重在训练过程中在这两个子空间上的投影分量占比，将数据保存为/root/data1/zjj/Grokking_Formulation/data/x+y/circuit_Competition.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/circuit_Competition.py，使用circuit_Competition.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc以及投影分量变化，横轴为训练步数，纵轴为acc和投影分量，双y轴，横轴依然采用对数刻度，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，执行上述任务。

训练已完成，实验数据已保存，结合之前的脚本，把所有数据全部重新收集一遍，分别保存到/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/下，然后执行之前的脚本，重新绘图，每个子图都要单独保存，在/root/data1/zjj/Grokking_Formulation/experiments/figures下，为每个模运算任务单独创建文件夹，分开保存子图，取消子图的标题和序号（a），配色要美观，字体配色也要统一，对于多分析大图，将其分割为若干子图单独保存，舍弃一个png/pdf文件中存在多张子图的形式。

# 2/2

## Compression

### Effective Rank
创建/root/data1/zjj/Grokking_Formulation/src/effective_rank.py，使用/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/checkpoints中保存的四个模运算的权重文件，对每层线性映射做SVD，并计算其谱熵和有效秩erank(W)=exp(H(W))，将数据分别保存为/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/effective_rank.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/effective_rank.py，使用/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/effective_rank.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc以及有效秩的变化，横轴为训练步数，纵轴为acc和有效秩，双y轴，横轴依然采用对数刻度，在/root/data1/zjj/Grokking_Formulation/experiments/figures下，为每个模运算任务单独创建文件夹，分开保存子图，取消子图的标题和序号（a），配色要美观，字体配色也要统一。

## Abstraction

### 参与率
PR用Gini和IPR稀疏代替，更合理。
### 环结构和群论表示
创建/root/data1/zjj/Grokking_Formulation/src/group_representation.py，使用/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/checkpoints中保存的四个模运算的权重文件，对输入嵌入We取出每个数对应的表示列Ex，对其做最小而成拟合：Rt​=argRmin​x∑​∥Ex+1​−REx​∥22​，然后使用相对残差定义ϵR​(t)=∑x​∥Ex+1​∥22​∑x​∥Ex+1​−Rt​Ex​∥22​​，，以及计算Rt是否正交：δorth​(t)=∥Rt⊤​Rt​−I∥F​，将残差和正交系数数据分别保存为/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/group_representation.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/group_representation.py，使用/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/group_representation.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc，相对残差以及δorth​(t)的变化，横轴为训练步数，纵轴为acc，ϵR​(t)和δorth​(t)，三y轴刻度，横轴依然采用对数刻度，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，为每个模运算任务单独创建文件夹，分开保存子图，取消子图的标题和序号（a），配色要美观，字体配色也要统一。
### 平坦度
原理：定义温度化能量：U(β)=Eθ∼pβ​(θ∣D)​[nLn​(θ)],pβ​(θ∣D)∝e−βnLn​(θ)π(θ)，自由能满足：∂Fn​(β)/∂β​=U(β)，在奇异值模型中，有效学习系数：^λeff​≈[Fn​(β2​)−Fn​(β1​)]/[log(β2​)−log(β1​)]​(在 Ln​(θ^)≈0 时),Fn​(β2​)−Fn​(β1​)=∫β1​β2​​U(β)dβ。
代码实现：在权重θt附近做小尺度随机扰动θt​+δ，用 MCMC / Langevin（或简化版的 importance sampling）在不同 β 下采样近似pβ，估计Ut​(β)进而计算有效λ。
创建/root/data1/zjj/Grokking_Formulation/src/flatness_slt.py，使用/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/checkpoints中保存的四个模运算的权重文件，计算其有效λ​，将λ数据分别保存为/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/flatness_slt.csv，然后创建/root/data1/zjj/Grokking_Formulation/src/plot/flatness_slt.py，使用/root/data1/zjj/Grokking_Formulation/data/x(+-×÷)y/flatness_slt.csv文件和metric.csv绘图体现随着训练步数，训练/测试的Acc，相对残差以及有效λ的变化，横轴为训练步数，纵轴为acc，λ，双y轴刻度，横轴依然采用对数刻度，最后保存/root/data1/zjj/Grokking_Formulation/experiments/figures下，为每个模运算任务单独创建文件夹，分开保存子图，取消子图的标题和序号（a），配色要美观，字体配色也要统一，对于多分析大图，将其分割为若干子图单独保存，舍弃一个png/pdf文件中存在多张子图的形式。



