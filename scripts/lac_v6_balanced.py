# -*- coding: utf-8 -*-
"""
lac v6 专项版 - 生肖均衡处理
"""
import pandas as pd
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

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

# 生肖均衡系数
balance_factor = {z: 5/4 if z == '马' else 1.0 for z in zodiacs}

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

def predict(data):
    scores = {z: 0 for z in zodiacs}
    
    # D1: 位置热度（均衡）
    pos_z = {p: Counter() for p in range(7)}
    for d in data[-30:]:
        for p in range(7):
            pos_z[p][d['zodiacs'][p]] += 1
    for p in range(7):
        for z, c in pos_z[p].most_common(3):
            scores[z] += c * (7-p) * 0.55 / balance_factor[z]
    
    # D2: 全局频率（均衡）
    fc = Counter()
    for d in data:
        fc.update(d['zodiacs'])
    for z in zodiacs:
        scores[z] += fc.get(z, 0) * 0.4 / balance_factor[z]
    
    # D5: 遗漏回补
    miss = {}
    for z in zodiacs:
        if z in data[-1]['zodiacs']:
            miss[z] = 0
        else:
            m = 1
            for i in range(len(data)-2, -1, -1):
                if z in data[i]['zodiacs']:
                    break
                m += 1
            miss[z] = m
    
    for z in zodiacs:
        if miss[z] >= 10: scores[z] += 18
        elif miss[z] >= 7: scores[z] += 15
        elif miss[z] >= 5: scores[z] += 11
        elif miss[z] >= 3: scores[z] += 7
    
    # D7: 近10期热度
    r10 = Counter()
    for d in data[-10:]:
        r10.update(d['zodiacs'])
    for z in zodiacs:
        h = r10.get(z, 0)
        if h >= 5: scores[z] += 6
        elif h >= 4: scores[z] += 5
        elif h >= 3: scores[z] += 4
        elif h >= 2: scores[z] += 2
    
    # D11: 重号
    for z in data[-1]['zodiacs']:
        scores[z] += 1.0
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == '__main__':
    lac_data = load_data(r'D:\Desktop\数据2027.xls', 9, 16)
    print(f"lac数据: {len(lac_data)}期")
    
    result = predict(lac_data)
    top3 = [z for z, s in result[:3]]
    top7 = [z for z, s in result[:7]]
    
    print(f"\n第87期预测:")
    print(f"TOP3: {top3}")
    print(f"TOP7: {top7}")
    print("排名: " + " ".join([f"{z}({s:.1f})" for z, s in result]))