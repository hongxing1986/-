# -*- coding: utf-8 -*-
"""105期预测的回测验证"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from collections import Counter

# 加载数据（1-104期）
df1 = pd.read_excel(r'G:\小元仓库\shengxiao_data_cloud.xlsx', header=None)
all_data = []
for idx, row in df1.iterrows():
    period = int(str(row.iloc[6]).replace('组', ''))
    nums = [int(row.iloc[7+i]) for i in range(7)]
    all_data.append({'period': period, 'nums': nums, 'last': nums[6], 'first': nums[0], 'sum': sum(nums)})

df2 = pd.read_excel(r'C:\Users\Lenovo\WPS Cloud Files\.1737354693\cachedata\DE7B48D710D14360AF004EECBCFB98FD\数据2026_correct.xlsx', header=None)
for idx, row in df2.iterrows():
    period = 87 + idx
    nums = [int(row.iloc[6+i]) for i in range(7)]
    all_data.append({'period': period, 'nums': nums, 'last': nums[6], 'first': nums[0], 'sum': sum(nums)})

for d in all_data:
    if d['period']==102: d['last']=20
    if d['period']==103: d['last']=6

# 追加104期
all_data.append({'period': 104, 'nums': [20,45,39,49,3,48,1], 'last': 1, 'first': 20, 'sum': 205})

print(f'数据: {len(all_data)}期\n')

# =====================================================
# 回测：lag1_zone=0规则
# =====================================================
print('='*60)
print('回测: lag1_zone=0 规则（前向验证）')
print('='*60)

# 从第10期开始验证
fh = 0
ft = 0
misses = []

for ti in range(10, len(all_data)):
    # ti是测试期的索引
    prev_last = all_data[ti-1]['last']
    prev_zone = (prev_last - 1) // 10
    
    if prev_zone != 0:
        continue
    
    # 训练集：ti之前的所有lag1_zone=0样本
    train = []
    for k in range(1, ti):
        if ((all_data[k]['last'] - 1) // 10) == 0:
            if all_data[k]['last'] != all_data[k-1]['last']:  # 排除重复
                train.append(all_data[k]['last'])
    
    if len(train) < 3:
        continue
    
    # TOP5
    c = Counter(train)
    top5 = [n for n, cnt in c.most_common(5)]
    
    # 实际结果
    actual = all_data[ti]['last']
    
    # 排除与上期相同
    if actual == prev_last:
        continue
    
    ft += 1
    hit = actual in top5
    if hit:
        fh += 1
    else:
        misses.append({
            'period': all_data[ti]['period'],
            'prev': prev_last,
            'actual': actual,
            'top5': top5
        })

print(f'验证结果: {fh}/{ft} = {fh/ft*100:.1f}%')
print(f'随机基准: 5/49 = 10.2%')
print(f'超额: {(fh/ft - 0.102)*100:.1f}%')

print(f'\n未命中 ({ft-fh}次):')
for m in misses:
    print(f'  {m["period"]}期: 上期{m["prev"]} → 实际{m["actual"]}, TOP5={m["top5"]}')

# =====================================================
# 回测：排除上期相同号 + 遗漏策略
# =====================================================
print('\n' + '='*60)
print('回测: 规则匹配 + 遗漏策略')
print('='*60)

# 综合策略：规则TOP5 + 遗漏>30期加权
def predict_with_strategy(data_list, test_idx):
    """预测test_idx期的号码"""
    prev_last = data_list[test_idx-1]['last']
    prev_zone = (prev_last - 1) // 10
    
    if prev_zone != 0:
        return None
    
    # 训练集
    train = []
    for k in range(1, test_idx):
        if ((data_list[k]['last'] - 1) // 10) == 0:
            if data_list[k]['last'] != data_list[k-1]['last']:
                train.append(data_list[k]['last'])
    
    if len(train) < 3:
        return None
    
    c = Counter(train)
    rule_top5 = [n for n, cnt in c.most_common(5)]
    
    # 计算遗漏（从test_idx-1往前算）
    appeared = set()
    for d in data_list[:test_idx]:
        appeared.add(d['last'])
    
    # 遗漏分
    missing_score = {}
    for n in range(1, 50):
        last_idx = None
        for i in range(test_idx-1, -1, -1):
            if data_list[i]['last'] == n:
                last_idx = i
                break
        if last_idx is None:
            missing_score[n] = 100  # 未出现过
        else:
            missing_score[n] = test_idx - 1 - last_idx
    
    # 综合得分
    scores = Counter()
    for n in rule_top5:
        scores[n] += 1
    
    for n, m in missing_score.items():
        if m > 30:
            scores[n] += 1
    
    # 排除上期相同
    scores[prev_last] = 0
    
    return [n for n, cnt in scores.most_common(5)]

# 执行回测
fh2 = 0
ft2 = 0
misses2 = []

for ti in range(10, len(all_data)):
    prev_zone = (all_data[ti-1]['last'] - 1) // 10
    if prev_zone != 0:
        continue
    
    # 排除重复
    if all_data[ti]['last'] == all_data[ti-1]['last']:
        continue
    
    pred = predict_with_strategy(all_data, ti)
    if pred is None:
        continue
    
    actual = all_data[ti]['last']
    ft2 += 1
    if actual in pred:
        fh2 += 1
    else:
        misses2.append({'period': all_data[ti]['period'], 'actual': actual, 'pred': pred})

print(f'验证结果: {fh2}/{ft2} = {fh2/ft2*100:.1f}%')

if ft2 > 0:
    print(f'\n未命中 ({ft2-fh2}次):')
    for m in misses2[:10]:
        print(f'  {m["period"]}期: 实际{m["actual"]}, 预测{m["pred"]}')

# =====================================================
# 随机基准对比
# =====================================================
print('\n' + '='*60)
print('基准对比')
print('='*60)

# 模拟随机选5个号
import random
random.seed(42)
hits = 0
trials = 0
for _ in range(10000):
    test_idx = random.randint(10, len(all_data)-1)
    prev_zone = (all_data[test_idx-1]['last'] - 1) // 10
    if prev_zone != 0:
        continue
    if all_data[test_idx]['last'] == all_data[test_idx-1]['last']:
        continue
    
    # 随机5个
    pred = random.sample(range(1, 50), 5)
    actual = all_data[test_idx]['last']
    if actual in pred:
        hits += 1
    trials += 1

print(f'随机基准: {hits}/{trials} = {hits/trials*100:.1f}%')
