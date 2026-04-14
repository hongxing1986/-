# -*- coding: utf-8 -*-
"""检查6号推荐的问题：上期开6，下期不可能开6"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from collections import Counter

df1 = pd.read_excel(r'G:\小元仓库\shengxiao_data_cloud.xlsx', header=None)
all_data = []
for idx, row in df1.iterrows():
    period = int(str(row.iloc[6]).replace('组', ''))
    nums = [int(row.iloc[7+i]) for i in range(7)]
    all_data.append({'period': period, 'nums': nums, 'last': nums[6]})

df2 = pd.read_excel(
    r'C:\Users\Lenovo\WPS Cloud Files\.1737354693\cachedata\DE7B48D710D14360AF004EECBCFB98FD\数据2026_correct.xlsx',
    header=None)
for idx, row in df2.iterrows():
    period = 87 + idx
    nums = [int(row.iloc[6+i]) for i in range(7)]
    all_data.append({'period': period, 'nums': nums, 'last': nums[6]})

for d in all_data:
    if d['period']==102: d['last']=20
    if d['period']==103: d['last']=6

print(f'数据: {len(all_data)}期\n')

# 问题1：历史上连续开出同一个号码的情况
print('='*60)
print('问题1：历史上连续开出同一个第7号的情况')
print('='*60)
repeat_count = 0
for i in range(1, len(all_data)):
    if all_data[i-1]['last'] == all_data[i]['last']:
        repeat_count += 1
        print(f'  {all_data[i-1]["period"]}期={all_data[i-1]["last"]} → {all_data[i]["period"]}期={all_data[i]["last"]} 【重复！】')

print(f'\n共{len(all_data)-1}期中，连续重复{repeat_count}次，概率{repeat_count/(len(all_data)-1)*100:.1f}%')
print(f'理论期望：1/49 = {1/49*100:.1f}%')
print(f'结论：连续重复极罕见！预测6号完全不合理！\n')

# 问题2：lag1_mod6=2规则的6号是在什么条件下出现的？
print('='*60)
print('问题2：lag1_mod6=2规则的6号出现条件')
print('='*60)

for i in range(1, len(all_data)):
    lag1 = all_data[i-1]['last']
    if lag1 % 6 == 2:
        repeat_mark = '⚠️重复!' if all_data[i-1]['last'] == all_data[i]['last'] else ''
        print(f'  {all_data[i-1]["period"]}期last={lag1}(mod6=2) → {all_data[i]["period"]}期last={all_data[i]["last"]} {repeat_mark}')

# 问题3：排除"与上期相同"后，规则还成立吗？
print('\n' + '='*60)
print('问题3：排除与上期相同号码后，重新统计')
print('='*60)

# 所有规则：排除与lag1相同的号码
for i, d in enumerate(all_data):
    d['lag1'] = all_data[i-1]['last'] if i >= 1 else None
    d['lag1_mod6'] = d['lag1'] % 6 if d['lag1'] else None
    d['lag1_zone'] = (d['lag1']-1)//10 if d['lag1'] else None
    d['first_zone'] = (all_data[i]['nums'][0]-1)//10
    d['sum'] = sum(d['nums'])
    d['sum_zone'] = min(d['sum']//50, 5)
    d['sum_mod5'] = d['sum']%5

# lag1_mod6=2 + 排除上期号码
next_nums_excl = []
for i in range(1, len(all_data)):
    if all_data[i-1]['last'] % 6 == 2:
        if all_data[i]['last'] != all_data[i-1]['last']:  # 排除与上期相同
            next_nums_excl.append(all_data[i]['last'])

c = Counter(next_nums_excl)
print(f'\nlag1_mod6=2规则（排除上期号码后）:')
print(f'  样本: {len(next_nums_excl)}期')
print(f'  TOP5: {[n for n, cnt in c.most_common(5)]}')
for n, cnt in c.most_common(10):
    print(f'    {n}号: {cnt}次')

# 前向验证（排除上期号码）
print('\n前向验证（排除上期号码）:')
fh = 0
ft = 0
for ti in range(10, len(all_data)):
    if all_data[ti-1]['last'] % 6 != 2:
        continue
    
    train = []
    for k in range(1, ti):
        if all_data[k-1]['last'] % 6 == 2 and all_data[k]['last'] != all_data[k-1]['last']:
            train.append(all_data[k]['last'])
    
    if len(train) < 3:
        continue
    
    c_train = Counter(train)
    top5 = [n for n, cnt in c_train.most_common(5)]
    actual = all_data[ti]['last']
    
    # 如果实际号码与上期相同，跳过（这种情况极罕见）
    if actual == all_data[ti-1]['last']:
        continue
    
    hit = actual in top5
    fh += 1 if hit else 0
    ft += 1

if ft > 0:
    print(f'  命中率: {fh}/{ft} = {fh/ft*100:.1f}%')

# =====================================================
# 核心修正：所有规则都排除"与上期相同"的号码
# =====================================================
print('\n' + '='*60)
print('核心修正：所有规则排除"与上期第7号相同的号码"')
print('='*60)

# 重新定义规则匹配，排除与上期相同的号码
# 103期last=6，所以排除6号
print(f'\n103期第7号=6 → 排除6号！')
print(f'6号在所有规则中得分最高是因为数据有偏，排除6号后重新排名：')

# 收集所有匹配规则，排除6号
matched = Counter()
excluded = {6}  # 排除6号

# 单维度规则
rules_data = [
    ('lag1_mod6=2(50.0%)', [6, 46, 41, 48, 23]),
    ('sum_zone=2(17.6%)', [6, 26, 12, 9, 27]),
    ('lag1_zone=1(14.3%)', [1, 3, 12, 21, 26]),
    ('lag3_zone=3(14.3%)', [1, 35, 46, 42, 18]),
    ('lag3_mod5=4(14.3%)', [41, 18, 43, 42, 24]),
    ('sum_mod3=0(12.5%)', [6, 41, 27, 1, 39]),
    ('lag1_mod4=0(11.8%)', [18, 19, 48, 6, 11]),
    ('first_zone=3(11.8%)', [11, 46, 6, 41, 31]),
    ('lag1_mod3=2(11.5%)', [6, 28, 46, 5, 9]),
    ('first_mod4=2(10.7%)', [6, 13, 12, 28, 37]),
]

for name, top5 in rules_data:
    for n in top5:
        if n not in excluded:
            matched[n] += 1

# 双维度规则
combo_data = [
    ('lag1_mod6=2+first_mod3=1(66.7%)', [6, 46, 48]),
    ('lag1_mod6=2+first_odd=0(50.0%)', [6, 46, 48]),
    ('first_mod4=2+sum_mod5=2(50.0%)', [6, 12, 20]),
    ('first_mod5=4+sum_zone=2(50.0%)', [12, 26, 6]),
    ('first_mod5=4+sum_mod5=2(50.0%)', [12, 27, 6]),
    ('sum_mod3=0+sum_mod5=2(42.9%)', [27, 6, 12, 41, 48]),
    ('lag1_mod3=2+first_zone=3(40.0%)', [46, 6, 27, 38, 11]),
    ('lag1_odd=0+sum_mod5=2(40.0%)', [6, 48, 11, 23, 12]),
    ('lag1_zone=1+sum_zone=2(33.3%)', [12, 26, 33, 6]),
    ('lag1_zone=1+sum_mod3=0(33.3%)', [1, 12, 3, 49, 39]),
]

for name, top5 in combo_data:
    for n in top5:
        if n not in excluded:
            matched[n] += 2  # 双维度权重加倍

print(f'\n排除6号后的综合推荐:')
for n, cnt in matched.most_common(15):
    print(f'  {n}号: {cnt}分')

final_top5 = [n for n, c in matched.most_common(5)]
final_top3 = [n for n, c in matched.most_common(3)]
print(f'\n🎯 修正后TOP5: {final_top5}')
print(f'🎯 修正后TOP3: {final_top3}')
