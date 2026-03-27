# -*- coding: utf-8 -*-
"""
号码预测系统 v2（优化版）
新增号码专属维度，提高预测准确率
"""
import pandas as pd
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ============ 加载数据 ============
fp = r'D:\Desktop\数据2026.xls'
df = pd.read_excel(fp, header=None)

all_data = []
for i in range(84):
    row = df.iloc[i]
    nums = []
    for col in range(7, 14):
        val = row[col]
        if pd.notna(val):
            s = str(val).strip()
            if s.isdigit():
                n = int(s)
                if 1 <= n <= 49:
                    nums.append(n)
    if len(nums) == 7:
        all_data.append({'numbers': nums})

all_data.append({'numbers': [14, 13, 27, 33, 6, 20, 19]})
print(f"✅ 数据: {len(all_data)}期")

# ============ 新增号码专属维度 ============
print("\n" + "="*80)
print("【号码预测 v2 - 新增维度】")
print("="*80)

print("""
📊 号码专属维度（新增加粗））

D2:  全局频率 - 数字历史出现次数
D5:  遗漏回补 - 遗漏周期分析
D6:  连开分析 - 数字连开规律
D7:  近10期热度 - 近期出现频率
D8:  冷号回补 - 遗漏超5期加分
D11: 重号 - 上期数字重复
D12: 冷热转换 - 冷热号调整
D13: 间隔周期 - 历史间隔分析

【新增号码专属维度】
N1:  奇偶比例 - 奇数/偶数分布预测
N2:  大小比例 - 大号(25+)/小号分布
N3:  和值分析 - 上期和值预测下期范围
N4:  尾数分布 - 0-9尾数出现规律
N5:  跨度分析 - 最大-最小号范围
N6:  AC值分析 - 数字复杂度
N7:  质合分析 - 质数/合数分布
N8:  连号分析 - 相邻号码规律
N9:  同尾分析 - 同尾数号码关联
N10: 区间分析 - 1-49分5区分析
""")

# ============ 优化版预测函数 ============
def predict_numbers_v2(data, top_n=1):
    scores = {n: 0.0 for n in range(1, 50)}
    
    # ====== 基础维度 D2, D5, D6, D7, D8, D11, D12, D13 ======
    
    # D2: 全局频率
    fc = Counter()
    for d in data:
        fc.update(d['numbers'])
    for n in range(1, 50):
        scores[n] += fc.get(n, 0) * 2.0
    
    # D5: 遗漏分析
    miss = {}
    for n in range(1, 50):
        if n in data[-1]['numbers']:
            miss[n] = 0
        else:
            m = 1
            for i in range(len(data)-2, -1, -1):
                if n in data[i]['numbers']: break
                m += 1
            miss[n] = m
    
    # 历史最大遗漏
    max_miss = {n: 0 for n in range(1, 50)}
    streak = {n: 0 for n in range(1, 50)}
    for d in data:
        for n in range(1, 50):
            if n in d['numbers']:
                if streak[n] > max_miss[n]: max_miss[n] = streak[n]
                streak[n] = 0
            else:
                streak[n] += 1
    
    for n in range(1, 50):
        if miss[n] > 0 and max_miss[n] > 0:
            ratio = miss[n] / max_miss[n]
            if ratio >= 0.85: scores[n] += 50
            elif ratio >= 0.70: scores[n] += 35
            elif ratio >= 0.50: scores[n] += 20
            elif ratio >= 0.30: scores[n] += 10
    
    # D6: 连开分析
    consec = {}
    for n in range(1, 50):
        if n in data[-1]['numbers']:
            c = 1
            for i in range(len(data)-2, -1, -1):
                if n in data[i]['numbers']: c += 1
                else: break
            consec[n] = c
        else:
            consec[n] = 0
    
    for n in range(1, 50):
        c = consec[n]
        if c >= 5: scores[n] -= 25
        elif c >= 4: scores[n] -= 15
        elif c >= 3: scores[n] -= 8
    
    # D7: 近10期热度
    r10 = Counter()
    for d in data[-10:]:
        r10.update(d['numbers'])
    for n in range(1, 50):
        heat = r10.get(n, 0)
        if heat >= 3: scores[n] += 20
        elif heat >= 2: scores[n] += 12
        elif heat >= 1: scores[n] += 5
        else: scores[n] += 3
    
    # D8: 冷号加分
    for n in range(1, 50):
        if miss.get(n, 0) >= 5: scores[n] += 12
    
    # D11: 重号
    last_nums = data[-1]['numbers']
    for n in last_nums: scores[n] += 15
    
    # D12: 冷热转换
    hot = [n for n, c in r10.items() if c >= 3]
    for n in hot: scores[n] += 5
    for n in range(1, 50):
        if r10.get(n, 0) <= 1 and miss.get(n, 0) >= 3: scores[n] += 15
    
    # D13: 间隔周期
    intervals = {n: [] for n in range(1, 50)}
    last = {n: -1 for n in range(1, 50)}
    for i, d in enumerate(data):
        for n in d['numbers']:
            if last[n] >= 0: intervals[n].append(i - last[n])
            last[n] = i
    for n in range(1, 50):
        if intervals[n]:
            avg_int = sum(intervals[n]) / len(intervals[n])
            if avg_int > 0 and miss.get(n, 0) > 0:
                if miss[n] >= avg_int - 1 and miss[n] <= avg_int + 2: scores[n] += 25
                elif miss[n] > avg_int: scores[n] += 12
    
    # ====== 新增号码专属维度 ======
    
    # N1: 奇偶比例分析
    last_odd = sum(1 for n in last_nums if n % 2 == 1)
    last_even = 7 - last_odd
    for n in range(1, 50):
        is_odd = n % 2 == 1
        if last_odd >= 5 and is_odd: scores[n] += 5  # 上期奇多，下期可能偶多
        elif last_even >= 5 and not is_odd: scores[n] += 5
    
    # N2: 大小比例分析
    last_big = sum(1 for n in last_nums if n > 25)
    last_small = 7 - last_big
    for n in range(1, 50):
        is_big = n > 25
        if last_big >= 5 and not is_big: scores[n] += 5  # 上期大号多，下期可能小号多
        elif last_small >= 5 and is_big: scores[n] += 5
    
    # N3: 和值分析
    last_sum = sum(last_nums)
    avg_sum = 175  # 历史平均和值
    for n in range(1, 50):
        if last_sum > avg_sum + 15 and n <= 25: scores[n] += 3  # 上期和值大，下期小号加分
        elif last_sum < avg_sum - 15 and n > 25: scores[n] += 3
    
    # N4: 尾数分析
    last_tails = [n % 10 for n in last_nums]
    tail_counts = Counter(last_tails)
    for n in range(1, 50):
        tail = n % 10
        if tail_counts.get(tail, 0) >= 2: scores[n] -= 5  # 同尾太多，扣分
        elif tail_counts.get(tail, 0) == 0: scores[n] += 3  # 同尾没出，加分
    
    # N5: 跨度分析
    last_min, last_max = min(last_nums), max(last_nums)
    last_span = last_max - last_min
    for n in range(1, 50):
        if last_span >= 35 and n < 15: scores[n] += 3  # 上期跨度大，下期小号加分
        elif last_span <= 20 and 15 <= n <= 35: scores[n] += 3
    
    # N7: 质合分析（质数:2,3,5,7,11,13,17,19,23,29,31,37,41,43,47）
    primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47]
    last_prime = sum(1 for n in last_nums if n in primes)
    last_composite = 7 - last_prime
    for n in range(1, 50):
        is_prime = n in primes
        if last_prime >= 5 and not is_prime: scores[n] += 3
        elif last_composite >= 5 and is_prime: scores[n] += 3
    
    # N8: 连号分析（相邻号码）
    last_sorted = sorted(last_nums)
    consecutive = 0
    for i in range(len(last_sorted)-1):
        if last_sorted[i+1] - last_sorted[i] == 1: consecutive += 1
    for n in range(1, 50):
        # 检查是否有相邻号码在上期出现
        has_adjacent = (n-1 in last_nums) or (n+1 in last_nums)
        if consecutive <= 1 and has_adjacent: scores[n] += 5  # 上期连号少，下期可能多
        elif consecutive >= 3 and not has_adjacent: scores[n] += 3  # 上期连号多，下期可能少
    
    # N10: 区间分析（1-49分5区）
    zones = [(1,10), (11,20), (21,30), (31,40), (41,49)]
    zone_counts = [0, 0, 0, 0, 0]
    for n in last_nums:
        for i, (z1, z2) in enumerate(zones):
            if z1 <= n <= z2:
                zone_counts[i] += 1
                break
    for n in range(1, 50):
        for i, (z1, z2) in enumerate(zones):
            if z1 <= n <= z2:
                if zone_counts[i] == 0: scores[n] += 8  # 上期某区没出
                elif zone_counts[i] >= 3: scores[n] -= 5  # 上期某区出太多
                break
    
    # N9: 同尾分析（历史同尾规律）
    tail_history = Counter()
    for d in data[-20:]:
        for n in d['numbers']:
            tail_history[n % 10] += 1
    for n in range(1, 50):
        tail = n % 10
        if tail_history.get(tail, 0) <= 3: scores[n] += 5  # 冷尾数加分
    
    # 排序
    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [n for n, s in result[:top_n]], result, miss, consec, r10

# ============ 回测验证 ============
print("\n" + "="*80)
print("【号码预测 v2 回测 - 第51-85期】")
print("="*80)

results = {1: [], 2: [], 3: []}
for i in range(50, 85):
    train = all_data[:i]
    actual = set(all_data[i]['numbers'])
    
    for n in [1, 2, 3]:
        pred, _, _, _, _ = predict_numbers_v2(train, top_n=n)
        hit = len(set(pred) & actual)
        results[n].append({'period': i+1, 'pred': pred, 'actual': actual, 'hit': hit})

# 统计
for n in [1, 2, 3]:
    total_hit = sum(r['hit'] for r in results[n])
    total_possible = n * 35
    theory = n / 49 * 100
    actual_pct = total_hit / total_possible * 100
    status = "✅ 超过理论!" if actual_pct > theory else f"❌ 低于理论(差{theory-actual_pct:.1f}%)"
    print(f"\n📊 预测{n}个号码: {total_hit}/{total_possible} = {actual_pct:.1f}%")
    print(f"   理论概率: {theory:.1f}%")
    print(f"   {status}")

# ============ 第86期预测 ============
print("\n" + "="*80)
print("【第86期 号码预测 v2】")
print("="*80)

top1, all_scores, miss, consec, r10 = predict_numbers_v2(all_data, top_n=1)
top2, _, _, _, _ = predict_numbers_v2(all_data, top_n=2)
top3, _, _, _, _ = predict_numbers_v2(all_data, top_n=3)
top5, _, _, _, _ = predict_numbers_v2(all_data, top_n=5)

print(f"\n📊 得分最高的10个号码:")
for rank, (n, s) in enumerate(all_scores[:10], 1):
    miss_info = f"漏{miss[n]}" if miss[n] > 0 else (f"连{consec[n]}" if consec[n] > 0 else "----")
    r10_info = f"近10期{r10.get(n,0)}次"
    print(f"  {rank}. {n:>2}号: {s:.1f}分  {miss_info}  {r10_info}")

print(f"\n🎯 预测结果:")
print(f"  TOP1: {top1[0]}号")
print(f"  TOP2: {top2}")
print(f"  TOP3: {top3}")
print(f"  TOP5: {top5}")
