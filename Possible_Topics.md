# Possible Topics

---

## Term Project 1

### 引言（Introduction）

本项目探索如何使用机器学习（ML）技术改进欧几里得空间染色数（chromatic number）的上下界，尤其关注 Hadwiger–Nelson 问题及其高维推广。

### Hadwiger–Nelson 问题（平面染色数）

令图  
\[
G = (V, E)
\]
定义如下：

\[
V = \{ x : x \in \mathbb{R}^2 \}, \quad 
E = \{ (v, v') : v, v' \in V,\ d(v, v') = 1 \}
\]

其中 \( d(v, v') \) 表示欧式距离。

**平面的染色数**（chromatic number of the plane）记为：

\[
\chi(\mathbb{R}^2)
\]

它表示将平面上所有点进行着色，使得所有距离为 1 的点颜色不同所需的最少颜色数。

**当前最优结果（SOTA）：**

\[
5 \le \chi(\mathbb{R}^2) \le 7
\]

---

### 三维空间的染色数 \( \chi(\mathbb{R}^3) \)

类似地，在三维空间定义：

\[
V = \{ x : x \in \mathbb{R}^3 \}, \quad 
E = \{ (v, v') : d(v, v') = 1 \}
\]

三维空间的染色数记为：

\[
\chi(\mathbb{R}^3)
\]

**当前最优结果（SOTA）：**

\[
6 \le \chi(\mathbb{R}^3) \le 15
\]

---

### 高维推广

上述问题可自然推广到更高维空间 \( \mathbb{R}^d \)（\( d \ge 4 \)），定义方式完全类比二维与三维情况。

---

### 渐近行为（Asymptotic Behavior）

当维度 \( d \to \infty \) 时，染色数 \( \chi(\mathbb{R}^d) \) 以指数级增长。  
已知的上下界形式如下：

\[
(1.239... + o(1))^d \le \chi(\mathbb{R}^d) \le (3 + o(1))^d
\]

---

### 机器学习目标

利用人工智能（或其他计算方法）改进上述任意维度 \( d \ge 2 \) 的染色数上下界（无论是提高下界还是降低上界）。

---

### 可行的机器学习方向

- 使用图神经网络（GNN）对有限子图进行分析  
- 用强化学习（RL）探索可行的着色策略  
- 使用监督学习分析已知配置及其染色数  
- 利用降维与模式识别技术探索高维着色结构 

### 文献
- A book about the problem: https://www.cs.umd.edu/~gasarch/COURSES/752/S22/mathcoloringbook.pdf
- polymath project：https://web.archive.org/web/20220216001534/https://asone.ai/polymath/index.php?title=Hadwiger-Nelson_problem

---

## Term Project II

### 使用大语言模型（LLM）进行数学发现

#### 参考论文
1. [DeepMind AlphaEvolve](https://arxiv.org/pdf/2506.13131)
2. [FunSearch](https://www.nature.com/articles/s41586-023-06924-6)

#### 方法总结

- 两篇论文的方法本质相似：使用大语言模型（LLM）生成用于求解数学问题的代码。
- 初始代码可能非常简单，也可能基于某种模板。
- 这些代码会被执行，产生结果；这些结果会作为反馈提供给 LLM。
- LLM 根据反馈进行反思（reflection）并优化，生成新的、更优代码。
  
---

### [Kissing number（接触数）](https://en.wikipedia.org/wiki/Kissing_number)

#### 定义

在 \(n\) 维空间中，问：最多有多少个 **不重叠的单位球**（半径为 1）可以同时“接触”一个中心单位球？

#### 条件

1. 存在一个中心单位球（半径 1），球心在原点。
2. 若干个单位球（半径同为 1）放置在周围。
3. 所有外围球必须与中心球相切（接触）。
4. 所有外围球之间不得重叠（最多允许相互接触）。

#### 目标

找到满足上述条件的**最大外围球数量**。这个最大数量即为该维度下的 **kissing number（接触数）**。

#### 不同维度的接触数

- **二维（2D）**：接触数为 **6**
- **三维（3D）**：接触数为 **12**
- **更高维度**：已知结果包括  
  - 4 维：24  
  - 8 维：240  
  - 24 维：196,560  

---

### AlphaEvolve 的贡献

- 论文将此问题视为一个构造(construction) 问题。
- AI 的任务是生成一段代码，这段代码的功能是生成外围球球心坐标，从而实现一种构造方案。
- AI 通过如下循环不断优化方案：  
  生成代码 → 执行并获得反馈 → 根据反馈修改代码。
- 在 **13 维空间**中，之前已知的构造最多能放置 592 个球；AlphaEvolve 通过其方法将这一数值提升到了 **593**。

