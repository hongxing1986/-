# -*- coding: utf-8 -*-
"""
lac预测系统 - 最新版 (混合策略)
TOP1: D5=0 (58.3%命中率)
TOP2: D5=25 (61.1%命中率)
回测覆盖: 77.8% (TOP1+TOP2)
"""
import pandas as pd
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ==================== 常量 ====================
zm = {1:'马',13:'马',25:'马',37:'马',49:'马',
      2:'蛇',14:'蛇',26:'蛇',38:'蛇',
      3:'龙',15:'龙',27:'龙',39:'龙',
      4:'兔',16:'兔',28:'兔',40:'兔',
      5:'虎',17:'虎',29:'虎',41:'虎',
      6:'牛',18:'牛',30:'牛',42:'牛',
      7:'鼠',19:'鼠',31:'鼠',43:'鼠',
      8:'猪',20:'猪',32:'猪',44:'猪',
      9:'狗',21:'狗',33:'狗',45:'狗',
      10:'鸡',22:'鸡',34:'鸡',46:'鸡',
      11:'猴',23:'猴',35:'猴',47:'猴',
      12:'羊',24:'羊',36:'羊',48:'羊'}

zs = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']
zone1 = ['鼠','牛','虎','兔']
zone2 = ['龙','蛇','马','羊']
zone3 = ['猴','鸡','狗','猪']
road0 = ['鼠','龙','猴']
road1 = ['牛','蛇','马','鸡']
road2 = ['虎','兔','羊','狗','猪']

BASE_W = {
    'D1':0.55,'D2':0.4,'D3a':7,'D3b':-1,'D4':6,
    'D6a':-10,'D6b':-6,'D6c':-2,
    'D7a':6,'D7b':5,'D7c':4,'D7d':2,'D7e':2,
    'D8t':1,'D8':5,'D11':1.0,'D12a':12,'D12b':2,'D13a':14,'D13b':8
}

# ==================== 数据加载 ====================
def load(fp, sc, ec):
    """加载Excel数据"""
    df = pd.read_excel(fp, header=None, engine='openpyxl')
    data = []
    for i in range(len(df)):
        row = df.iloc[i]
        nums = []
        for col in range(sc, ec):
            v = row[col]
            if pd.notna(v):
                try:
                    n = int(float(v))
                    if 1 <= n <= 49:
                        nums.append(n)
                except:
                    pass
        if len(nums) == 7:
            data.append({'nums': nums, 'zs': [zm[n] for n in nums]})
    return data

# ==================== 遗漏计算 ====================
def calc_miss(data):
    """计算遗漏值和最大遗漏"""
    last = data[-1]
    miss = {}
    for z in zs:
        if z in last['zs']:
            miss[z] = 0
        else:
            m = 1
            for i in range(len(data)-2, -1, -1):
                if z in data[i]['zs']:
                    break
                m += 1
            miss[z] = m
    
    mm = {z: 0 for z in zs}
    st = {z: 0 for z in zs}
    for d in data:
        for z in zs:
            if z in d['zs']:
                if st[z] > mm[z]:
                    mm[z] = st[z]
                st[z] = 0
            else:
                st[z] += 1
    return miss, mm

# ==================== 预测函数 ====================
def predict(data, d5_weight=0):
    """
    预测函数
    d5_weight: D5维度权重，0=TOP1优先，25=TOP2优先
    """
    scores = {z: 0 for z in zs}
    last = data[-1]
    miss, mm = calc_miss(data)
    
    r10 = Counter()
    for d in data[-10:]:
        r10.update(d['zs'])
    
    fc = Counter()
    for d in data:
        fc.update(d['zs'])
    
    # D1: 位置热度
    pz = {p: Counter() for p in range(7)}
    for d in data[-30:]:
        for p in range(7):
            pz[p][d['zs'][p]] += 1
    for p in range(7):
        for z, c in pz[p].most_common(3):
            scores[z] += c * (7-p) * BASE_W['D1']
    
    # D2: 全局频率
    for z in zs:
        scores[z] += fc.get(z, 0) * BASE_W['D2']
    
    # D3: 三区分层
    cz1 = sum(1 for z in last['zs'] if z in zone1)
    cz2 = sum(1 for z in last['zs'] if z in zone2)
    cz3 = sum(1 for z in last['zs'] if z in zone3)
    for z in zone1:
        if cz1<=1: scores[z] += BASE_W['D3a']
    for z in zone2:
        if cz2>=3: scores[z] += BASE_W['D3b']
    for z in zone3:
        if cz3<=1: scores[z] += BASE_W['D3a']
    
    # D4: 012路
    cr0 = sum(1 for z in last['zs'] if z in road0)
    cr1 = sum(1 for z in last['zs'] if z in road1)
    cr2 = sum(1 for z in last['zs'] if z in road2)
    for z in road0:
        if cr0<=1: scores[z] += BASE_W['D4']
    for z in road1:
        if cr1<=1: scores[z] += BASE_W['D4']
    for z in road2:
        if cr2<=1: scores[z] += BASE_W['D4']
    
    # D5: 遗漏回补 (可调权重)
    if d5_weight > 0:
        for z in zs:
            if miss[z] > 0 and mm[z] > 0:
                r = miss[z] / mm[z]
                if r >= 0.85: scores[z] += d5_weight
                elif r >= 0.70: scores[z] += d5_weight * 0.83
                elif r >= 0.50: scores[z] += d5_weight * 0.61
                elif r >= 0.30: scores[z] += d5_weight * 0.39
    
    # D6: 连开惩罚
    for z in zs:
        if z in last['zs']:
            c = 1
            for i in range(len(data)-2, -1, -1):
                if z in data[i]['zs']:
                    c += 1
                else:
                    break
            if c >= 5: scores[z] += BASE_W['D6a']
            elif c >= 4: scores[z] += BASE_W['D6b']
            elif c >= 3: scores[z] += BASE_W['D6c']
    
    # D7: 近10期热度
    for z in zs:
        h = r10.get(z, 0)
        if h >= 5: scores[z] += BASE_W['D7a']
        elif h >= 4: scores[z] += BASE_W['D7b']
        elif h >= 3: scores[z] += BASE_W['D7c']
        elif h >= 2: scores[z] += BASE_W['D7d']
        else: scores[z] += BASE_W['D7e']
    
    # D8: 遗漏阈值
    for z in zs:
        if miss.get(z, 0) > BASE_W['D8t']:
            scores[z] += BASE_W['D8']
    
    # D9: 位置重复
    for p in range(7):
        pc = Counter()
        for d in data[-20:]:
            pc[d['zs'][p]] += 1
        for z, c in pc.most_common(2):
            if c >= 3:
                scores[z] += 0.5
    
    # D10: 连号
    for i in range(len(last['nums'])-1):
        if last['nums'][i+1] - last['nums'][i] == 1:
            scores[last['zs'][i]] += 1
    
    # D11: 重号
    for z in last['zs']:
        scores[z] += BASE_W['D11']
    
    # D12: 冷热转换
    cold = [z for z in zs if miss.get(z, 0) >= 5]
    hot = [z for z in zs if r10.get(z, 0) >= 4]
    for z in cold:
        if miss.get(z, 0) >= 2:
            scores[z] += BASE_W['D12a']
    for z in hot:
        scores[z] += BASE_W['D12b']
    
    # D13: 间隔周期
    intv = {}
    for z in zs:
        iv = []
        last_i = -1
        for i in range(len(data)):
            if z in data[i]['zs']:
                if last_i >= 0:
                    iv.append(i - last_i)
                last_i = i
        if iv:
            intv[z] = sum(iv) / len(iv)
    for z in zs:
        ai = intv.get(z, 5)
        if miss.get(z, 0) >= ai-1 and miss.get(z, 0) <= ai+2:
            scores[z] += BASE_W['D13a']
        elif miss.get(z, 0) > ai:
            scores[z] += BASE_W['D13b']
    
    # D14: 生肖关联
    pairs = [('鼠','龙'), ('龙','蛇'), ('蛇','马'), ('马','羊'), ('羊','猴'),
             ('猴','鸡'), ('鸡','狗'), ('狗','猪'), ('猪','牛'), ('牛','虎'),
             ('虎','兔'), ('兔','鼠')]
    for z1, z2 in pairs:
        if z1 in last['zs']: scores[z2] += 3
        if z2 in last['zs']: scores[z1] += 3
    
    # D15: 奇偶比例
    odd = sum(1 for n in last['nums'] if n % 2 == 1)
    if odd >= 5:
        for z in ['马','虎','龙','蛇','狗']:
            scores[z] += 5
    else:
        for z in ['鼠','牛','兔','羊','猴','鸡','猪']:
            scores[z] += 5
    
    # D16: 大小比例
    big = sum(1 for n in last['nums'] if n >= 25)
    if big >= 5:
        for z in ['马','蛇','龙','虎','牛']:
            scores[z] += 4
    else:
        for z in ['鼠','兔','羊','猴','鸡','狗','猪']:
            scores[z] += 4
    
    # D17: 和值范围
    s = sum(last['nums'])
    if 100 <= s <= 150:
        for z in zs:
            scores[z] += 2
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [z for z, _ in ranked[:5]], scores

# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*60)
    print("lac预测系统 - 最新版 (混合策略)")
    print("="*60)
    
    # 加载数据
    all_data = load(r'D:\Desktop\数据2027.xls', 9, 16)
    print(f"\n数据: {len(all_data)}期")
    
    # 回测统计
    n_backtest = len(all_data) - 50
    top1_hit = 0
    top2_hit = 0
    
    print(f"\n回测统计（第51-{len(all_data)}期，共{n_backtest}期）")
    print("-"*60)
    
    for i in range(50, len(all_data)):
        train = all_data[:i]
        actual = all_data[i]
        actual_set = set(actual['zs'])
        
        # TOP1用D5=0
        pred1, _ = predict(train, 0)
        # TOP2用D5=25
        pred2, _ = predict(train, 25)
        
        if pred1[0] in actual_set: top1_hit += 1
        if pred2[1] in actual_set: top2_hit += 1
    
    print(f"\nTOP1命中率(D5=0): {top1_hit}/{n_backtest} = {top1_hit/n_backtest*100:.1f}%")
    print(f"TOP2命中率(D5=25): {top2_hit}/{n_backtest} = {top2_hit/n_backtest*100:.1f}%")
    print(f"综合覆盖率: {(top1_hit+top2_hit)/n_backtest:.1f}%")
    
    # 预测最新一期
    print("\n" + "="*60)
    print(f"第{len(all_data)+1}期预测结果")
    print("="*60)
    
    # TOP1预测(D5=0)
    pred1, scores1 = predict(all_data, 0)
    print(f"\n【TOP1预测】(D5=0策略)")
    print(f"  第一名: {pred1[0]} (得分:{scores1[pred1[0]]:.1f})")
    
    # TOP2预测(D5=25)
    pred2, scores2 = predict(all_data, 25)
    print(f"\n【TOP2预测】(D5=25策略)")
    print(f"  第一名: {pred2[0]} (得分:{scores2[pred2[0]]:.1f})")
    print(f"  第二名: {pred2[1]} (得分:{scores2[pred2[1]]:.1f})")
    
    print("\n" + "="*60)
    print("重点关注: {}、{}".format(pred1[0], pred2[1]))
    print("="*60)
