# -*- coding: utf-8 -*-
"""
xac & lac 13维度权重优化系统
通过多轮回测找出最佳权重配置
"""
import pandas as pd
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

zodiac_map = {
    12:'羊', 24:'羊', 36:'羊', 48:'羊',
    11:'猴', 23:'猴', 35:'猴', 47:'猴',
    10:'鸡', 22:'鸡', 34:'鸡', 46:'鸡',
    9:'狗', 21:'狗', 33:'狗', 45:'狗',
    8:'猪', 20:'猪', 32:'猪', 44:'猪',
    7:'鼠', 19:'鼠', 31:'鼠', 43:'鼠',
    6:'牛', 18:'牛', 30:'牛', 42:'牛',
    5:'虎', 17:'虎', 29:'虎', 41:'虎',
    4:'兔', 16:'兔', 28:'兔', 40:'兔',
    3:'龙', 15:'龙', 27:'龙', 39:'龙',
    2:'蛇', 14:'蛇', 26:'蛇', 38:'蛇',
    1:'马', 13:'马', 25:'马', 37:'马', 49:'马'
}
zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
zone1 = ['鼠', '牛', '虎', '兔']
zone2 = ['龙', '蛇', '马', '羊']
zone3 = ['猴', '鸡', '狗', '猪']
road0 = ['鼠', '龙', '猴']
road1 = ['牛', '蛇', '马', '鸡']
road2 = ['虎', '兔', '羊', '狗', '猪']

# ============ 预测函数 ============
def predict_13d(data, w):
    scores = {z: 0.0 for z in zodiacs}

    # D1
    pos_z = {p: Counter() for p in range(7)}
    for d in data[-30:]:
        for p in range(7): pos_z[p][d['zodiacs'][p]] += 1
    for p in range(7):
        for z, c in pos_z[p].most_common(3):
            scores[z] += c * (7-p) * w['D1']

    # D2
    fc = Counter()
    for d in data: fc.update(d['zodiacs'])
    for z in zodiacs: scores[z] += fc.get(z, 0) * w['D2']

    # D3
    cz1 = sum(1 for z in data[-1]['zodiacs'] if z in zone1)
    cz2 = sum(1 for z in data[-1]['zodiacs'] if z in zone2)
    cz3 = sum(1 for z in data[-1]['zodiacs'] if z in zone3)
    for z in zodiacs:
        iz = 1 if z in zone1 else (2 if z in zone2 else 3)
        if iz==1 and cz1<=1: scores[z] += w['D3a']
        elif iz==2 and cz2>=3: scores[z] += w['D3b']
        elif iz==3 and cz3<=1: scores[z] += w['D3a']

    # D4
    r0 = sum(1 for z in data[-1]['zodiacs'] if z in road0)
    r1 = sum(1 for z in data[-1]['zodiacs'] if z in road1)
    r2 = sum(1 for z in data[-1]['zodiacs'] if z in road2)
    for z in zodiacs:
        if z in road0 and r0<=1: scores[z] += w['D4']
        elif z in road1 and r1<=1: scores[z] += w['D4']
        elif z in road2 and r2<=1: scores[z] += w['D4']

    # D5
    miss = {}
    for z in zodiacs:
        if z in data[-1]['zodiacs']: miss[z] = 0
        else:
            m = 1
            for i in range(len(data)-2, -1, -1):
                if z in data[i]['zodiacs']: break
                m += 1
            miss[z] = m
    max_m = {z: 0 for z in zodiacs}
    st = {z: 0 for z in zodiacs}
    for d in data:
        for z in zodiacs:
            if z in d['zodiacs']:
                if st[z] > max_m[z]: max_m[z] = st[z]
                st[z] = 0
            else: st[z] += 1
    for z in zodiacs:
        if miss[z] > 0 and max_m[z] > 0:
            r = miss[z] / max_m[z]
            if r >= 0.85: scores[z] += w['D5a']
            elif r >= 0.70: scores[z] += w['D5b']
            elif r >= 0.50: scores[z] += w['D5c']
            elif r >= 0.30: scores[z] += w['D5d']

    # D6
    consec = {}
    for z in zodiacs:
        if z in data[-1]['zodiacs']:
            c = 1
            for i in range(len(data)-2, -1, -1):
                if z in data[i]['zodiacs']: c += 1
                else: break
            consec[z] = c
        else: consec[z] = 0
    for z in zodiacs:
        c = consec[z]
        if c >= 5: scores[z] += w['D6a']
        elif c >= 4: scores[z] += w['D6b']
        elif c >= 3: scores[z] += w['D6c']

    # D7
    r10 = Counter()
    for d in data[-10:]: r10.update(d['zodiacs'])
    for z in zodiacs:
        h = r10.get(z, 0)
        if h >= 5: scores[z] += w['D7a']
        elif h >= 4: scores[z] += w['D7b']
        elif h >= 3: scores[z] += w['D7c']
        elif h >= 2: scores[z] += w['D7d']
        else: scores[z] += w['D7e']

    # D8
    for z in zodiacs:
        if miss[z] > w['D8t']: scores[z] += w['D8']

    # D9
    pc = {}
    for p in range(7):
        pc[p] = Counter()
        for d in data[-20:]: pc[p][d['zodiacs'][p]] += 1
    for z in zodiacs:
        tot = sum(0.5 for p in range(7) if pc[p].get(z, 0) >= 2)
        if tot > 0: scores[z] += tot

    # D10
    lc = sum(1 for d in data for j in range(len(d['zodiacs'])-1)
        if (zodiacs.index(d['zodiacs'][j+1]) - zodiacs.index(d['zodiacs'][j])) % 12 == 1)
    lp = lc / len(data)
    for z in zodiacs:
        zi = zodiacs.index(z)
        for pz in data[-1]['zodiacs']:
            if abs(zodiacs.index(pz) - zi) % 12 == 1:
                scores[z] += 2 * lp * 10; break

    # D11
    rec = sum(len(set(data[i-1]['zodiacs']) & set(data[i]['zodiacs'])) for i in range(1, len(data)))
    ra = rec / (len(data)-1)
    for z in zodiacs:
        if z in data[-1]['zodiacs']: scores[z] += ra * w['D11']

    # D12
    zh = {z: r10.get(z, 0) for z in zodiacs}
    hot = [z for z, c in zh.items() if c >= 5]
    cold = [z for z, c in zh.items() if c <= 1]
    for z in zodiacs:
        if z in cold and miss.get(z, 0) >= 2: scores[z] += w['D12a']
        elif z in hot: scores[z] += w['D12b']

    # D13
    iv = {z: [] for z in zodiacs}
    lt = {z: -1 for z in zodiacs}
    for i, d in enumerate(data):
        for z in d['zodiacs']:
            if lt[z] >= 0: iv[z].append(i - lt[z])
            lt[z] = i
    for z in zodiacs:
        if iv[z]:
            ai = sum(iv[z]) / len(iv[z])
            if ai > 0:
                if miss.get(z, 0) >= ai-1 and miss.get(z, 0) <= ai+2: scores[z] += w['D13a']
                elif miss.get(z, 0) > ai: scores[z] += w['D13b']

    result = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [z for z, s in result[:3]], miss, consec

# ============ 回测函数 ============
def backtest(data, w):
    total = 0
    for i in range(50, 85):
        top3, _, _ = predict_13d(data[:i], w)
        actual = set(data[i]['zodiacs'])
        hit = len(actual & set(top3))
        total += hit
    return total / 105 * 100

# ============ 加载数据 ============
# xac数据
df_xac = pd.read_excel(r'D:\Desktop\数据2026.xls', header=None)
xac_data = []
for i in range(84):
    row = df_xac.iloc[i]
    nums = []
    for col in range(7, 14):
        val = row[col]
        if pd.notna(val):
            s = str(val).strip()
            if s.isdigit():
                n = int(s)
                if 1 <= n <= 49: nums.append(n)
    if len(nums) == 7:
        xac_data.append({'numbers': nums, 'zodiacs': [zodiac_map[n] for n in nums]})
xac_data.append({'numbers': [14,13,27,33,6,20,19], 'zodiacs': [zodiac_map[n] for n in [14,13,27,33,6,20,19]]})

# lac数据
df_lac = pd.read_excel(r'D:\Desktop\数据2027.xls', header=None)
lac_data = []
for i in range(85):
    row = df_lac.iloc[i]
    nums = []
    for col in range(9, 16):
        val = row[col]
        if pd.notna(val):
            s = str(val).strip()
            if s.isdigit():
                n = int(s)
                if 1 <= n <= 49: nums.append(n)
    if len(nums) == 7:
        lac_data.append({'numbers': nums, 'zodiacs': [zodiac_map[n] for n in nums]})

print("="*80)
print("       【xac & lac 权重优化系统】")
print("="*80)

# ============ xac优化 ============
print("\n【xac 优化中...】")
best_xac = 50.5
best_w_xac = None

# 测试多组权重
weight_sets_xac = [
    # 基准组（原优化版）
    {'D1':0.5,'D2':0.35,'D3a':6,'D3b':-2,'D4':5,'D5a':15,'D5b':12,'D5c':8,'D5d':4,
     'D6a':-12,'D6b':-8,'D6c':-4,'D7a':8,'D7b':6,'D7c':4,'D7d':2,'D7e':1,
     'D8t':2,'D8':4,'D11':1.2,'D12a':10,'D12b':3,'D13a':12,'D13b':6},
    # 提高遗漏权重
    {'D1':0.55,'D2':0.4,'D3a':7,'D3b':-1,'D4':6,'D5a':18,'D5b':15,'D5c':10,'D5d':6,
     'D6a':-10,'D6b':-6,'D6c':-3,'D7a':7,'D7b':5,'D7c':3,'D7d':2,'D7e':1,
     'D8t':1,'D8':5,'D11':1.0,'D12a':12,'D12b':4,'D13a':14,'D13b':8},
    # 提高热号惯性
    {'D1':0.5,'D2':0.35,'D3a':6,'D3b':-2,'D4':5,'D5a':15,'D5b':12,'D5c':8,'D5d':4,
     'D6a':-10,'D6b':-6,'D6c':-3,'D7a':10,'D7b':8,'D7c':6,'D7d':3,'D7e':1,
     'D8t':2,'D8':4,'D11':1.2,'D12a':10,'D12b':5,'D13a':12,'D13b':6},
    # 平衡型
    {'D1':0.52,'D2':0.38,'D3a':6.5,'D3b':-1.5,'D4':5.5,'D5a':16,'D5b':13,'D5c':9,'D5d':5,
     'D6a':-11,'D6b':-7,'D6c':-3,'D7a':8,'D7b':6,'D7c':4,'D7d':2,'D7e':1,
     'D8t':2,'D8':4.5,'D11':1.1,'D12a':11,'D12b':4,'D13a':13,'D13b':7},
    # 遗漏优先
    {'D1':0.45,'D2':0.4,'D3a':7,'D3b':-1,'D4':6,'D5a':20,'D5b':16,'D5c':12,'D5d':8,
     'D6a':-8,'D6b':-5,'D6c':-2,'D7a':6,'D7b':5,'D7c':4,'D7d':2,'D7e':2,
     'D8t':1,'D8':6,'D11':1.0,'D12a':14,'D12b':3,'D13a':15,'D13b':8},
]

for i, w in enumerate(weight_sets_xac):
    acc = backtest(xac_data, w)
    print(f"  方案{i+1}: {acc:.1f}%")
    if acc > best_xac:
        best_xac = acc
        best_w_xac = w.copy()

print(f"\n✅ xac 最佳准确率: {best_xac:.1f}%")

# ============ lac优化 ============
print("\n【lac 优化中...】")
best_lac = 47.6
best_w_lac = None

weight_sets_lac = [
    # 基准组
    {'D1':0.5,'D2':0.35,'D3a':6,'D3b':-2,'D4':5,'D5a':15,'D5b':12,'D5c':8,'D5d':4,
     'D6a':-12,'D6b':-8,'D6c':-4,'D7a':8,'D7b':6,'D7c':4,'D7d':2,'D7e':1,
     'D8t':2,'D8':4,'D11':1.2,'D12a':10,'D12b':3,'D13a':12,'D13b':6},
    # 提高龙鸡鼠权重（最常遗漏）
    {'D1':0.55,'D2':0.4,'D3a':7,'D3b':-1,'D4':6,'D5a':18,'D5b':15,'D5c':11,'D5d':7,
     'D6a':-10,'D6b':-6,'D6c':-2,'D7a':6,'D7b':5,'D7c':4,'D7d':2,'D7e':2,
     'D8t':1,'D8':5,'D11':1.0,'D12a':12,'D12b':2,'D13a':14,'D13b':8},
    # 遗漏优先
    {'D1':0.45,'D2':0.4,'D3a':7,'D3b':-1,'D4':6,'D5a':20,'D5b':16,'D5c':12,'D5d':8,
     'D6a':-8,'D6b':-5,'D6c':-2,'D7a':6,'D7b':5,'D7c':4,'D7d':2,'D7e':2,
     'D8t':1,'D8':6,'D11':1.0,'D12a':14,'D12b':3,'D13a':15,'D13b':8},
    # 平衡型
    {'D1':0.52,'D2':0.38,'D3a':6.5,'D3b':-1.5,'D4':5.5,'D5a':16,'D5b':13,'D5c':9,'D5d':5,
     'D6a':-11,'D6b':-7,'D6c':-3,'D7a':8,'D7b':6,'D7c':4,'D7d':2,'D7e':1,
     'D8t':2,'D8':4.5,'D11':1.1,'D12a':11,'D12b':3,'D13a':13,'D13b':7},
]

for i, w in enumerate(weight_sets_lac):
    acc = backtest(lac_data, w)
    print(f"  方案{i+1}: {acc:.1f}%")
    if acc > best_lac:
        best_lac = acc
        best_w_lac = w.copy()

print(f"\n✅ lac 最佳准确率: {best_lac:.1f}%")

# ============ 预测第86期 ============
print("\n" + "="*80)
print("       【第86期 优化后预测结果】")
print("="*80)

# xac预测
top3_xac, miss_xac, consec_xac = predict_13d(xac_data, best_w_xac)
print(f"\n【xac 第86期预测】准确率: {best_xac:.1f}%")
print(f"  TOP3: {top3_xac}")

# lac预测
top3_lac, miss_lac, consec_lac = predict_13d(lac_data, best_w_lac)
print(f"\n【lac 第86期预测】准确率: {best_lac:.1f}%")
print(f"  TOP3: {top3_lac}")

print("\n" + "="*80)
print("【优化总结】")
print("="*80)
print(f"  xac: {best_xac:.1f}% (优化后)")
print(f"  lac: {best_lac:.1f}% (优化后)")
