# -*- coding: utf-8 -*-
"""
xac v5 终极版 - 完整17维度 + 生肖均衡
"""
import pandas as pd
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 生肖映射表
zodiac_map = {
    1: '马', 13: '马', 25: '马', 37: '马', 49: '马',
    2: '蛇', 14: '蛇', 26: '蛇', 38: '蛇',
    3: '龙', 15: '龙', 27: '龙', 39: '龙',
    4: '兔', 16: '兔', 28: '兔', 40: '兔',
    5: '虎', 17: '虎', 29: '虎', 41: '虎',
    6: '牛', 18: '牛', 30: '牛', 42: '牛',
    7: '鼠', 19: '鼠', 31: '鼠', 43: '鼠',
    8: '猪', 20: '猪', 32: '猪', 44: '猪',
    9: '狗', 21: '狗', 33: '狗', 45: '狗',
    10: '鸡', 22: '鸡', 34: '鸡', 46: '鸡',
    11: '猴', 23: '猴', 35: '猴', 47: '猴',
    12: '羊', 24: '羊', 36: '羊', 48: '羊'
}
zodiacs = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
zone1 = ['鼠', '牛', '虎', '兔']
zone2 = ['龙', '蛇', '马', '羊']
zone3 = ['猴', '鸡', '狗', '猪']
road0 = ['鼠', '龙', '猴']
road1 = ['牛', '蛇', '马', '鸡']
road2 = ['虎', '兔', '羊', '狗', '猪']

# 均衡系数
balance_factor = {z: 5/4 if z == '马' else 1.0 for z in zodiacs}

# 基础权重
BASE_W = {
    'D1':0.55,'D2':0.4,'D3a':7,'D3b':-1,'D4':6,'D5a':18,'D5b':15,'D5c':11,'D5d':7,
    'D6a':-10,'D6b':-6,'D6c':-2,'D7a':6,'D7b':5,'D7c':4,'D7d':2,'D7e':2,
    'D8t':1,'D8':5,'D11':1.0,'D12a':12,'D12b':2,'D13a':14,'D13b':8,
}

def load_data(filepath, start_col, end_col):
    df = pd.read_excel(filepath, header=None)
    data = []
    for i in range(len(df)):
        row = df.iloc[i]
        nums = []
        for col in range(start_col, end_col):
            val = row[col]
            if pd.notna(val):
                s = str(val).strip()
                if s.isdigit():
                    n = int(s)
                    if 1 <= n <= 49: nums.append(n)
        if len(nums) == 7:
            data.append({'numbers': nums, 'zodiacs': [zodiac_map[n] for n in nums]})
    return data

def predict(data, w=BASE_W):
    scores = {z: 0.0 for z in zodiacs}
    
    # D1: 位置热度
    pos_z = {p: Counter() for p in range(7)}
    for d in data[-30:]:
        for p in range(7):
            pos_z[p][d['zodiacs'][p]] += 1
    for p in range(7):
        for z, c in pos_z[p].most_common(3):
            scores[z] += c * (7-p) * w['D1']
    
    # D2: 全局频率
    fc = Counter()
    for d in data:
        fc.update(d['zodiacs'])
    for z in zodiacs:
        scores[z] += fc.get(z, 0) * w['D2']
    
    # D3: 三区分层
    cz1 = sum(1 for z in data[-1]['zodiacs'] if z in zone1)
    cz2 = sum(1 for z in data[-1]['zodiacs'] if z in zone2)
    cz3 = sum(1 for z in data[-1]['zodiacs'] if z in zone3)
    for z in zodiacs:
        iz = 1 if z in zone1 else (2 if z in zone2 else 3)
        if iz==1 and cz1<=1: scores[z] += w['D3a']
        elif iz==2 and cz2>=3: scores[z] += w['D3b']
        elif iz==3 and cz3<=1: scores[z] += w['D3a']
    
    # D4: 012路
    r0 = sum(1 for z in data[-1]['zodiacs'] if z in road0)
    r1 = sum(1 for z in data[-1]['zodiacs'] if z in road1)
    r2 = sum(1 for z in data[-1]['zodiacs'] if z in road2)
    for z in zodiacs:
        if z in road0 and r0<=1: scores[z] += w['D4']
        elif z in road1 and r1<=1: scores[z] += w['D4']
        elif z in road2 and r2<=1: scores[z] += w['D4']
    
    # D5: 遗漏回补
    miss = {}
    for z in zodiacs:
        if z in data[-1]['zodiacs']:
            miss[z] = 0
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
    
    # D6: 连开惩罚
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
    
    # D7: 近10期热度
    r10 = Counter()
    for d in data[-10:]:
        r10.update(d['zodiacs'])
    for z in zodiacs:
        h = r10.get(z, 0)
        if h >= 5: scores[z] += w['D7a']
        elif h >= 4: scores[z] += w['D7b']
        elif h >= 3: scores[z] += w['D7c']
        elif h >= 2: scores[z] += w['D7d']
        else: scores[z] += w['D7e']
    
    # D8: 遗漏阈值
    for z in zodiacs:
        if miss.get(z, 0) > w['D8t']: scores[z] += w['D8']
    
    # D9: 位置重复
    pc = {}
    for p in range(7):
        pc[p] = Counter()
        for d in data[-20:]:
            pc[p][d['zodiacs'][p]] += 1
    for z in zodiacs:
        tot = sum(0.5 for p in range(7) if pc[p].get(z, 0) >= 2)
        scores[z] += tot
    
    # D10: 连号
    for i in range(len(data[-1]['numbers'])-1):
        if data[-1]['numbers'][i+1] - data[-1]['numbers'][i] == 1:
            z = data[-1]['zodiacs'][i]
            scores[z] += 1
    
    # D11: 重号
    for z in data[-1]['zodiacs']:
        scores[z] += w['D11']
    
    # D12: 冷热转换
    cold = [z for z in zodiacs if miss.get(z, 0) >= 5]
    hot = [z for z in zodiacs if r10.get(z, 0) >= 4]
    for z in zodiacs:
        if z in cold and miss.get(z, 0) >= 2: scores[z] += w['D12a']
        elif z in hot: scores[z] += w['D12b']
    
    # D13: 间隔周期
    intervals = {}
    for z in zodiacs:
        intv = []
        last = -1
        for i in range(len(data)):
            if z in data[i]['zodiacs']:
                if last >= 0:
                    intv.append(i - last)
                last = i
        if intv:
            intervals[z] = sum(intv) / len(intv)
    for z in zodiacs:
        ai = intervals.get(z, 5)
        if miss.get(z, 0) >= ai-1 and miss.get(z, 0) <= ai+2: scores[z] += w['D13a']
        elif miss.get(z, 0) > ai: scores[z] += w['D13b']
    
    # D14: 生肖关联性
    zodiac_pairs = [
        ('鼠','龙'), ('龙','蛇'), ('蛇','马'), ('马','羊'), ('羊','猴'),
        ('猴','鸡'), ('鸡','狗'), ('狗','猪'), ('猪','牛'), ('牛','虎'),
        ('虎','兔'), ('兔','鼠')
    ]
    last_z = data[-1]['zodiacs']
    for z1, z2 in zodiac_pairs:
        if z1 in last_z:
            scores[z2] += 3
        if z2 in last_z:
            scores[z1] += 3
    
    # D15: 奇偶比例
    last_nums = data[-1]['numbers']
    odd = sum(1 for n in last_nums if n % 2 == 1)
    even = 7 - odd
    if odd >= 5:
        for z in ['马','虎','龙','蛇','狗']:
            scores[z] += 5
    elif even >= 5:
        for z in ['鼠','牛','兔','羊','猴','鸡','猪']:
            scores[z] += 5
    
    # D16: 大小号比例
    big = sum(1 for n in last_nums if n >= 25)
    small = 7 - big
    if big >= 5:
        for z in ['马','蛇','龙','虎','牛']:
            scores[z] += 4
    elif small >= 5:
        for z in ['鼠','兔','羊','猴','鸡','狗','猪']:
            scores[z] += 4
    
    # D17: 和值范围
    s = sum(data[-1]['numbers'])
    if 100 <= s <= 150:
        for z in zodiacs:
            scores[z] += 2
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# 主程序
if __name__ == '__main__':
    xac_data = load_data(r'D:\Desktop\数据2026.xls', 7, 14)
    print(f"xac数据: {len(xac_data)}期")
    
    result = predict(xac_data)
    top3 = [z for z, s in result[:3]]
    top7 = [z for z, s in result[:7]]
    
    print(f"\n第87期预测 (17维度完整版):")
    print(f"TOP3: {top3}")
    print(f"TOP7: {top7}")
    print("\n完整排名:")
    for z, s in result:
        print(f"  {z}: {s:.1f}")
