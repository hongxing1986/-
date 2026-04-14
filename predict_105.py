# -*- coding: utf-8 -*-
"""105期预测 - 基于104期结果"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from collections import Counter

# 加载数据
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

# 修正102-103期
for d in all_data:
    if d['period']==102: d['last']=20
    if d['period']==103: d['last']=6

# 追加104期
all_data.append({
    'period': 104,
    'nums': [20, 45, 39, 49, 3, 48, 1],
    'last': 1,
    'first': 20,
    'sum': sum([20, 45, 39, 49, 3, 48, 1])
})

print(f'数据: {len(all_data)}期 (1-104)\n')

# 104期特征
p104 = all_data[-1]
p103 = all_data[-2]

print('='*60)
print('104期数据分析')
print('='*60)
print(f'第7号: {p104["last"]}')
print(f'第1号: {p104["first"]}')
print(f'总和: {p104["sum"]}')
print(f'上期(103期)第7号: {p103["last"]}')

# 计算104期的特征
lag1 = p103['last']  # 103期第7号 = 6
lag2 = all_data[-3]['last']  # 102期第7号 = 20
lag3 = all_data[-4]['last']  # 101期第7号 = 39

lag1_zone = (lag1 - 1) // 10  # 6在0区间(1-10)
lag1_mod6 = lag1 % 6
lag1_mod5 = lag1 % 5
lag1_mod4 = lag1 % 4
lag1_mod3 = lag1 % 3
lag1_digit = lag1 % 10

lag2_zone = (lag2 - 1) // 10  # 20在1区间(11-20)
lag2_mod5 = lag2 % 5

first_zone = (p104['first'] - 1) // 10  # 20在1区间(11-20)
sum_zone = min(p104['sum'] // 50, 5)  # 205//50=4

print(f'\n104期特征:')
print(f'  lag1={lag1}, lag1_zone={lag1_zone}, lag1_mod6={lag1_mod6}, lag1_mod5={lag1_mod5}')
print(f'  lag2={lag2}, lag2_zone={lag2_zone}')
print(f'  first={p104["first"]}, first_zone={first_zone}')
print(f'  sum={p104["sum"]}, sum_zone={sum_zone}')

# =====================================================
# 规则匹配：lag1_zone=0（103期第7号在1-10区间）
# =====================================================
print('\n' + '='*60)
print('规则匹配: lag1_zone=0')
print('='*60)

# 历史上lag1_zone=0的所有期
samples = []
for i in range(1, len(all_data)):
    prev_last = all_data[i-1]['last']
    prev_zone = (prev_last - 1) // 10
    if prev_zone == 0:  # 上期第7号在1-10区间
        samples.append({
            'period': all_data[i]['period'],
            'lag1': prev_last,
            'last': all_data[i]['last']
        })

print(f'历史样本 ({len(samples)}期):')
for s in samples:
    excl = '⚠️重复' if s['lag1'] == s['last'] else ''
    print(f'  {s["period"]}期: 上期{s["lag1"]}(区间0) → 开{s["last"]} {excl}')

# 统计
nums = [s['last'] for s in samples]
c = Counter(nums)
print(f'\n号码分布:')
for n, cnt in c.most_common(10):
    print(f'  {n}号: {cnt}次')

top5_all = [n for n, cnt in c.most_common(5)]
print(f'\n全量TOP5: {top5_all}')

# 前向验证（排除上期相同号码）
print('\n前向验证（排除上期相同号码）:')
fh = 0
ft = 0
for ti in range(10, len(all_data)-1):  # 不包含104期
    prev_last = all_data[ti]['last']
    prev_zone = (prev_last - 1) // 10
    if prev_zone != 0:
        continue
    
    # 训练集
    train = []
    for k in range(1, ti):
        if ((all_data[k]['last'] - 1) // 10) == 0:
            if all_data[k]['last'] != all_data[k-1]['last']:
                train.append(all_data[k]['last'])
    
    if len(train) < 3:
        continue
    
    c_train = Counter(train)
    top5 = [n for n, cnt in c_train.most_common(5)]
    actual = all_data[ti+1]['last']
    
    # 排除与上期相同
    if actual == all_data[ti]['last']:
        continue
    
    hit = actual in top5
    if hit: fh += 1
    ft += 1

if ft > 0:
    print(f'  命中率: {fh}/{ft} = {fh/ft*100:.1f}%')
else:
    print('  样本不足')

# =====================================================
# 其他规则
# =====================================================
print('\n' + '='*60)
print('其他规则')
print('='*60)

# lag1_mod6 = 6%6 = 0
print('\nlag1_mod6=0:')
samples_mod6 = []
for i in range(1, len(all_data)):
    if all_data[i-1]['last'] % 6 == 0:
        samples_mod6.append(all_data[i]['last'])
c6 = Counter(samples_mod6)
print(f'  样本: {len(samples_mod6)}期')
print(f'  TOP5: {[n for n,c in c6.most_common(5)]}')

# lag1_mod5 = 6%5 = 1
print('\nlag1_mod5=1:')
samples_mod5 = []
for i in range(1, len(all_data)):
    if all_data[i-1]['last'] % 5 == 1:
        if all_data[i]['last'] != all_data[i-1]['last']:
            samples_mod5.append(all_data[i]['last'])
c5 = Counter(samples_mod5)
print(f'  样本: {len(samples_mod5)}期')
print(f'  TOP5: {[n for n,c in c5.most_common(5)]}')

# =====================================================
# 综合推荐
# =====================================================
print('\n' + '='*60)
print('105期综合推荐')
print('='*60)

# 排除104期开出的号码：1号
exclude = {1}

matched = Counter()

# lag1_zone=0规则
for n in top5_all:
    if n not in exclude:
        matched[n] += 1

# lag1_mod6=0规则
for n, cnt in c6.most_common(5):
    if n not in exclude:
        matched[n] += 1

# lag1_mod5=1规则
for n, cnt in c5.most_common(5):
    if n not in exclude:
        matched[n] += 1

print(f'\n排除1号后综合排名:')
for n, cnt in matched.most_common(15):
    print(f'  {n}号: {cnt}分')

top5_final = [n for n, c in matched.most_common(5)]
top3_final = [n for n, c in matched.most_common(3)]

print(f'\n🎯 最终TOP5: {top5_final}')
print(f'🎯 TOP3: {top3_final}')

# =====================================================
# 遗漏值策略
# =====================================================
print('\n' + '='*60)
print('遗漏值策略')
print('='*60)

# 统计所有号码在1-104期的出现次数
all_nums = set(range(1, 50))
appeared = set()
for d in all_data:
    appeared.add(d['last'])

# 遗漏最大的号码
missing = all_nums - appeared
print(f'\n104期内从未出现的号码: {sorted(missing)}')

# 计算每个号码的遗漏期数
last_appear = {}
for n in range(1, 50):
    for i in range(len(all_data)-1, -1, -1):
        if all_data[i]['last'] == n:
            last_appear[n] = len(all_data) - i - 1  # 距离104期的期数
            break

# 按遗漏期数排序
sorted_missing = sorted(last_appear.items(), key=lambda x: x[1] if x[1] else 999, reverse=True)
print(f'\n遗漏排名（前15）:')
for n, miss in sorted_missing[:15]:
    print(f'  {n}号: 遗漏{miss}期')

# 遗漏50-100期的号码
mid_missing = [n for n, m in last_appear.items() if 30 <= m <= 100]
print(f'\n遗漏30-100期: {sorted(mid_missing)}')

# 最终推荐结合遗漏策略
print(f'\n🎯 综合推荐（含遗漏策略）:')
# 遗漏号码给更高权重
for n in sorted(missing):
    matched[n] += 3  # 未出现过的号码权重更高

for n in mid_missing:
    matched[n] += 1

final_top5 = [n for n, c in matched.most_common(5)]
final_top3 = [n for n, c in matched.most_common(3)]

print(f'\n最终TOP5: {final_top5}')
print(f'TOP3: {final_top3}')
