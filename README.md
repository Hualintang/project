利用 Python 进行 NBA 比赛数据分析
===
内容简介<br>
---
根据NBA比赛的以往数据预测接下来某场比赛的结果。我们将基于 2017-2018 年的 NBA 常规赛及季后赛的比赛统计数据，预测 2018-2019 常规赛每场赛事的结果。<br>

![image](https://github.com/Hualintang/hualintang/blob/master/python/timg.jpeg)

实验知识点
---
·NBA 球队的Elo score计算<br>·特征向量<br>·逻辑回归<br>

实验流程
---
按照下面的流程实现 NBA 比赛数据分析的任务：<br>1.获取比赛统计数据<br>2.比赛数据分析，得到代表每场比赛每支队伍状态的特征表达<br>3.利用机器学习方法学习每场比赛与胜利队伍的关系，并对 2018-2019 的比赛进行预测

获取NBA比赛数据
---

比赛数据介绍
---
在本次实验中，我们将采用 Basketball Reference.com 中的统计数据。在这个网站中，你可以看到不同球员、队伍、赛季和联盟比赛的基本统计数据，如得分、犯规次数、胜负次数等情况。而我们在这里将会使用 2017-2018 NBA Season Summary 中的数据

![image](https://github.com/Hualintang/hualintang/blob/master/python/bs_re.png)

在这个 2017-2018 总结的所有表格中，我们将使用的是以下三个数据表格：<br>
Team Per Game Stats：每支队伍平均每场比赛的表现统计

| 数据名| 含义  |
| ----- | ----- |
| 内容1 | 内容2 |
|Rk -- Rank|	排名|
|G -- Games|参与的比赛场数（都为 82 场）|
|MP -- Minutes| Played	平均每场比赛进行的时间|
|FG--Field Goals|	投球命中次数|
|FGA--Field Goal Attempts|	投射次数|
|FG%--Field Goal Percentage|	投球命中次数|
|3P--3-Point Field Goals|	三分球命中次数|
|3PA--3-Point Field Goal Attempts|	三分球投射次数|
|3P%--3-Point Field Goal Percentage|	三分球命中率|
|2P--2-Point Field Goals|	二分球命中次数|
|2PA--2-point Field Goal Attempts|	二分球投射次数|
|2P%--2-Point Field Goal Percentage|	二分球命中率|
|FT--Free Throws|	罚球命中次数|
|FTA--Free Throw Attempts|	罚球投射次数|
|FT%--Free Throw Percentage|	罚球命中率|
|ORB--Offensive Rebounds|	进攻篮板球|
|DRB--Defensive Rebounds|	防守篮板球|
|TRB--Total Rebounds|	篮板球总数|
|AST--Assists|	助攻|
|STL--Steals|	抢断|
|BLK -- Blocks|	封盖|
|TOV -- Turnovers|	失误|
|PF -- Personal Fouls|	个犯|
|PTS -- Points|	得分|

Opponent Per Game Stats：所遇到的对手平均每场比赛的统计信息，所包含的统计数据与 Team Per Game Stats 中的一致，只是代表的是该球队对应的对手的统计信息

Miscellaneous Stats：综合统计数据

|数据名|含义|
| ----- | ----- |
|Rk (Rank)|	排名|
|Age|	队员的平均年龄|
|W (Wins)|	胜利次数|
|L (Losses)|	失败次数|
|PW (Pythagorean wins)|	基于毕达哥拉斯理论计算的赢的概率|
|PL (Pythagorean losses)|	基于毕达哥拉斯理论计算的输的概率|
|MOV (Margin of Victory)|	赢球次数的平均间隔|
|SOS (Strength of Schedule)|	用以评判对手选择与其球队或是其他球队的难易程度对比，0 为平均线，可以为正负数|
|SRS (Simple Rating System)|	简易评级系统，根据他们的积分差异对球队进行排名|
|ORtg (Offensive Rating)|	每 100 个比赛回合中的进攻比例|
|DRtg (Defensive Rating)|	每 100 个比赛回合中的防守比例|
|Pace (Pace Factor)|	每 48 分钟内大概会进行多少个回合|
|FTr (Free Throw Attempt Rate)|	罚球次数所占投射次数的比例|
|3PAr (3-Point Attempt Rate)|	三分球投射占投射次数的比例|
|TS% (True Shooting Percentage)|	二分球、三分球和罚球的总共命中率|
|eFG% (Effective Field Goal Percentage)|	有效的投射百分比（含二分球、三分球）|
|TOV% (Turnover Percentage)|	每 100 场比赛中失误的比例|
|ORB% (Offensive Rebound Percentage)|	球队中平均每个人的进攻篮板的比例|
|FT/FGA|	罚球所占投射的比例|
|eFG% (Opponent Effective Field Goal Percentage)|	对手投射命中比例|
|TOV% (Opponent Turnover Percentage)|	对手的失误比例|
|DRB% (Defensive Rebound Percentage)|	球队平均每个球员的防守篮板比例|
|FT/FGA (Opponent Free Throws Per Field Goal Attempt)|	对手的罚球次数占投射次数的比例|

毕达哥拉斯定律
---
![image](https://github.com/Hualintang/hualintang/blob/master/python/bdgls.png)

我们将用这三个表格来评估球队过去的战斗力，另外还需 2017-2018 NBA Schedule and Results 中的 2017~2018 年的 NBA 常规赛及季后赛的每场比赛的比赛数据，用以评估Elo score。在Basketball Reference.com中按照从常规赛至季后赛的时间，列出了 2017 年 10 月份至 2018 年 6 月份的每场比赛的比赛情况。

![image](https://github.com/Hualintang/hualintang/blob/master/python/saicheng.png)

可在上图中看到 2017 年 10 月份的部分比赛数据。在每个 Schedule 表格中所包含的数据为

|数据名|含义|
|---|---|
|Date|	比赛日期|
|Start (ET)|	比赛开始时间|
|Visitor/Neutral|	客场作战队伍|
|PTS|	客场队伍最后得分|
|Home/Neutral|	主场队伍|
|PTS|	主场队伍最后得分|
|Notes|	备注，表明是否为加时赛等|

在预测时，我们同样也需要在 2018-2019 NBA Schedule and Results 中 2018~2019 年的 NBA 的常规赛比赛安排数据。

获取比赛数据
---

我们将以获取 Team Per Game Stats 表格数据为例，展示如何获取这三项统计数据：

1.进入到 Basketball Reference.com 中，在导航栏中选择Season并选择2017~2018赛季中的Summary：

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc1.png)

2.进入到 2017~2018 年的Summary界面后，滑动窗口找到Team Per Game Stats表格，并选择左上方的 Share & more，在其下拉菜单中选择 Get table as CSV (for Excel)：

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc2.png)

3.复制在界面中生成的 csv 格式数据，并粘贴至一个文本编辑器保存为 csv 文件即可：

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc3.png)

数据分析
---

在获取到数据之后，我们将利用每支队伍过去的比赛情况和 Elo 等级分来判断每支比赛队伍的可胜概率。在评价到每支队伍过去的比赛情况时，将使用到 Team Per Game Stats、Opponent Per Game Stats 和 Miscellaneous Stats（之后简称为 T、O 和 M 表）这三个表格的数据，作为代表比赛中某支队伍的比赛特征。我们的目标是实现针对每场比赛，预测比赛中哪支队伍最终将会获胜，但并不是给出绝对的胜败情况，而是预判胜利的队伍有多大的获胜概率。因此我们将建立一个代表比赛的特征向量。由两支队伍的以往比赛统计情况（T、O 和Ｍ表）和两个队伍各自的 Elo 等级分构成。

在这里我们将基于国际象棋比赛，大致地介绍下 Elo 等级划分制度。在上图中 Eduardo 在窗户上写下的公式就是根据Logistic Distribution计算 PK 双方（A 和 B）对各自的胜率期望值计算公式。假设 A 和 B 的当前等级分为 RAR_ARA和 RBR_BRB，则 A 对 B 的胜率期望值为：

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc4.png)

B 对 A 的胜率期望值为

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc5.png)

如果棋手 A 在比赛中的真实得分 SAS_ASA（胜 1 分，和 0.5 分，负 0 分）和他的胜率期望值 EAE_AEA不同，则他的等级分要根据以下公式进行调整：

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc6.png)

在国际象棋中，根据等级分的不同 K 值也会做相应的调整：

·大于等于2400，K=16<br>
·2100~2400 分，K=24<br>
·小于等于2100，K=32<br>

因此我们将会用以表示某场比赛数据的特征向量为（假如 A 与 B 队比赛）：[A 队 Elo score, A 队的 T,O 和 M 表统计数据，B 队 Elo score, B 队的 T,O 和 M 表统计数据]

基于数据进行模型训练和预测
---

实验前期准备
---

在本次实验环境中，我们将会使用到 python 的pandas，numpy，scipy和sklearn库，接下来，我们下载相应的数据文件并解压。


