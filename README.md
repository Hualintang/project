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

美國棒球統計專家比爾˙詹姆斯在80年代初整理美國職業網球聯盟球隊的過去成績時，發現可以用一支球隊的總得分和總失分算出勝率。然後用直角三角型斜線長的平方，等於其他兩邊乘和的“畢達哥拉斯定理”算出了一個公式。就是“勝率＝總得分的平方÷（總得分的平方＋總失分的平方）”，即“畢達哥拉斯乘率”。

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

注：ELO等级分制度（英语：Elo rating system）是指由匈牙利裔美国物理学家Arpad Elo创建的一个衡量各类对弈活动水平的评价方法，是当今对弈水平评估的公认的权威方法。被广泛用于国际象棋、围棋、足球、篮球等运动。

假设棋手A和B的当前等级分分别为RA和RB，则按Logistic distribution A对B的胜率期望值当为：

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc4.png)

B 对 A 的胜率期望值为

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc5.png)

假如棋手A在比赛中的真实得分SA（胜=1分，和=0.5分，负=0分）和他的胜率期望值EA不同，需要根据以下公式进行调整：

![image](https://github.com/Hualintang/hualintang/blob/master/python/lc6.png)

在国际象棋中，根据等级分的不同 K 值也会做相应的调整：

·评分> 2400: K = 16<br>
·2100 < 评分 < 2400: k = 24;<br>
·评分<2100: K = 32<br>

因此我们将会用以表示某场比赛数据的特征向量为（假如 A 与 B 队比赛）：[A 队 Elo score, A 队的 T,O 和 M 表统计数据，B 队 Elo score, B 队的 T,O 和 M 表统计数据]

基于数据进行模型训练和预测
---

实验前期准备
---

在本次实验环境中，我们将会使用到 python 的pandas，numpy，scipy和sklearn库，接下来，我们解压相应的数据文件。

在data文件夹中，包含了 2017~2018 年的 NBA 数据 T,O 和 M 表，及经处理后的常规赛和挑战赛的比赛数据2017-2018_result.csv，这个数据文件是我们通过在basketball-reference.com的 2017-2018 Schedule and result 的几个月份比赛数据中提取得到的，其中包括三个字段：<br>

·WTeam: 比赛胜利队伍<br>·LTeam: 失败队伍<br>·WLoc: 胜利队伍一方所在的为主场或是客场<br>

另外一个文件就是18-19Schedule.csv，也是经过我们加工处理得到的 NBA 在 2018-2019 年的常规赛的比赛安排，其中包括两个字段：

·Vteam: 客场作战队伍<br>·Hteam: 主场作战队伍

代码实现
---

解压完实验数据后，就可以正式开始实验了。

首先，引入实验相关模块：

```
import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model#线性回归算法模型
from sklearn.model_selection import cross_val_score#交叉验证
```

设置回归训练时所需用到的参数变量：

```
# 当每支队伍没有elo等级分时，赋予其基础elo等级分
base_elo = 1600
team_elos = {} 
team_stats = {}
X = []
y = []
# 存放数据的目录
folder = 'data'
```

在最开始需要初始化数据，从 T、O 和 M 表格中读入数据，去除一些无关数据并将这三个表格通过Team属性列进行连接:

```
# 根据每支队伍的Miscellaneous Opponent，Team统计数据csv文件进行初始化
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')
    return team_stats1.set_index('Team', inplace=False, drop=True)
```

获取每支队伍的Elo Score等级分函数，当在开始没有等级分时，将其赋予初始base_elo值：

```
def get_elo(team):
    try:
        return team_elos[team]
    except:
        # 当最初没有elo时，给每个队伍最初赋base_elo
        team_elos[team] = base_elo
        return team_elos[team]
```

定义计算每支球队的Elo等级分函数：

```
# 计算每个球队的elo值
def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff  * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    # 根据rank级别修改K值
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    
    # 更新 rank 数值
    new_winner_rank = round(winner_rank + (k * (1 - odds)))      
    new_loser_rank = round(loser_rank + (k * (0 - odds)))
    return new_winner_rank, new_loser_rank
```

基于我们初始好的统计数据，及每支队伍的 Elo score 计算结果，建立对应 2017~2018 年常规赛和季后赛中每场比赛的数据集（在主客场比赛时，我们认为主场作战的队伍更加有优势一点，因此会给主场作战队伍相应加上 100 等级分）：

```
def  build_dataSet(all_data):
    print("Building data set..")
    X = []
    skip = 0
    for index, row in all_data.iterrows():

        Wteam = row['WTeam']
        Lteam = row['LTeam']

        #获取最初的elo或是每个队伍最初的elo值
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        # 给主场比赛的队伍加上100的elo值
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        # 把elo当为评价每个队伍的第一个特征值
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # 添加我们从basketball reference.com获得的每个队伍的统计信息
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)

        # 将两支队伍的特征值随机的分配在每场比赛数据的左右两侧
        # 并将对应的0/1赋给y值
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        if skip == 0:
            print('X',X)
            skip = 1

        # 根据这场比赛的数据更新队伍的elo值
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(X), y
```

最终在 main 函数中调用这些数据处理函数，使用 sklearn 的Logistic Regression方法建立回归模型：

```
if __name__ == '__main__':

    Mstat = pd.read_csv(folder + '/17-18Miscellaneous_Stat.csv')
    Ostat = pd.read_csv(folder + '/17-18Opponent_Per_Game_Stat.csv')
    Tstat = pd.read_csv(folder + '/17-18Team_Per_Game_Stat.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)

    result_data = pd.read_csv(folder + '/2017-2018_result.csv')
    X, y = build_dataSet(result_data)

    # 训练网络模型
    print("Fitting on %d game samples.." % len(X))

    model = linear_model.LogisticRegression()
    model.fit(X, y)

    # 利用10折交叉验证计算训练正确率
    print("Doing cross-validation..")
    print(cross_val_score(model, X, y, cv = 10, scoring='accuracy', n_jobs=-1).mean())
```

最终利用训练好的模型在 18~19 年的常规赛数据中进行预测。

利用模型对一场新的比赛进行胜负判断，并返回其胜利的概率：

```
def predict_winner(team_1, team_2, model):
    features = []

    # team 1，客场队伍
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)

    # team 2，主场队伍
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)

    features = np.nan_to_num(features)
    return model.predict_proba([features])
```

在 main 函数中调用该函数，并将预测结果输出到18-19Result.csv文件中：

```
# 利用训练好的model在18-19年的比赛中进行预测

print('Predicting on new schedule..')
schedule1617 = pd.read_csv(folder + '/18-19Schedule.csv')
result = []
for index, row in schedule1819.iterrows():
    team1 = row['Vteam']
    team2 = row['Hteam']
    pred = predict_winner(team1, team2, model)
    prob = pred[0][0]
    if prob > 0.5:
        winner = team1
        loser = team2
        result.append([winner, loser, prob])
    else:
        winner = team2
        loser = team1
        result.append([winner, loser, 1 - prob])

with open('18-19Result.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['win', 'lose', 'probability'])
    writer.writerows(result)
    print('done.')
```

最后，我们实验 Pandas 预览生成预测结果文件18-19Result.csv文件：

```
pd.read_csv('18-19Result.csv',header=0)
```

实验总结
---
利用`Basketball-reference.com`的部分统计数据，计算每支 NBA 比赛队伍的`Elo socre`，和利用这些基本统计数据评价每支队伍过去的比赛情况，并且根据国际等级划分方法Elo Score对队伍现在的战斗等级进行评分，最终结合这些不同队伍的特征判断在一场比赛中，哪支队伍能够占到优势。
