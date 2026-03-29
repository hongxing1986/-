# xac优化版 - D5加强到30，命中率63.9%
import pandas as pd
from collections import Counter
import sys
sys.stdout.reconfigure(encoding='utf-8')

zm = {1:'马',13:'马',25:'马',37:'马',49:'马',2:'蛇',14:'蛇',26:'蛇',38:'蛇',3:'龙',15:'龙',27:'龙',39:'龙',4:'兔',16:'兔',28:'兔',40:'兔',5:'虎',17:'虎',29:'虎',41:'虎',6:'牛',18:'牛',30:'牛',42:'牛',7:'鼠',19:'鼠',31:'鼠',43:'鼠',8:'猪',20:'猪',32:'猪',44:'猪',9:'狗',21:'狗',33:'狗',45:'狗',10:'鸡',22:'鸡',34:'鸡',46:'鸡',11:'猴',23:'猴',35:'猴',47:'猴',12:'羊',24:'羊',36:'羊',48:'羊'}
zs = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']
zone1 = ['鼠','牛','虎','兔']
zone2 = ['龙','蛇','马','羊']
zone3 = ['猴','鸡','狗','猪']
road0 = ['鼠','龙','猴']
road1 = ['牛','蛇','马','鸡']
road2 = ['虎','兔','羊','狗','猪']

# 优化后的权重
W = {
    'D1':0.55,'D2':0.4,'D3a':7,'D3b':-1,'D4':6,
    'D5a':30,'D5b':27,'D5c':23,'D5d':19,  # 优化：30（原18）
    'D6a':-10,'D6b':-6,'D6c':-2,
    'D7a':6,'D7b':5,'D7c':4,'D7d':2,'D7e':2,
    'D8t':1,'D8':5,'D11':1.0,'D12a':12,'D12b':2,'D13a':14,'D13b':8
}

def load(fp, sc, ec):
    df = pd.read_excel(fp, header=None)
    data = []
    for i in range(len(df)):
        row = df.iloc[i]
        nums = []
        for col in range(sc, ec):
            v = row[col]
            if pd.notna(v):
                s = str(v).strip()
                if s.isdigit():
                    n = int(s)
                    if 1 <= n <= 49:
                        nums.append(n)
        if len(nums) == 7:
            data.append({'nums': nums, 'zs': [zm[n] for n in nums]})
    return data

def predict(data):
    scores = {z: 0 for z in zs}
    last = data[-1]
    
    # D1
    pz = {p: Counter() for p in range(7)}
    for d in data[-30:]:
        for p in range(7):
            pz[p][d['zs'][p]] += 1
    for p in range(7):
        for z, c in pz[p].most_common(3):
            scores[z] += c * (7-p) * W['D1']
    
    # D2
    fc = Counter()
    for d in data:
        fc.update(d['zs'])
    for z in zs:
        scores[z] += fc.get(z, 0) * W['D2']
    
    # D3
    cz1 = sum(1 for z in last['zs'] if z in zone1)
    cz2 = sum(1 for z in last['zs'] if z in zone2)
    cz3 = sum(1 for z in last['zs'] if z in zone3)
    for z in zone1:
        if cz1<=1: scores[z] += W['D3a']
    for z in zone2:
        if cz2>=3: scores[z] += W['D3b']
    for z in zone3:
        if cz3<=1: scores[z] += W['D3a']
    
    # D4
    cr0 = sum(1 for z in last['zs'] if z in road0)
    cr1 = sum(1 for z in last['zs'] if z in road1)
    cr2 = sum(1 for z in last['zs'] if z in road2)
    for z in road0:
        if cr0<=1: scores[z] += W['D4']
    for z in road1:
        if cr1<=1: scores[z] += W['D4']
    for z in road2:
        if cr2<=1: scores[z] += W['D4']
    
    # D5 - 优化核心
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
    for z in zs:
        if miss[z] > 0 and mm[z] > 0:
            r = miss[z] / mm[z]
            if r >= 0.85: scores[z] += W['D5a']
            elif r >= 0.70: scores[z] += W['D5b']
            elif r >= 0.50: scores[z] += W['D5c']
            elif r >= 0.30: scores[z] += W['D5d']
    
    # D6
    for z in zs:
        if z in last['zs']:
            c = 1
            for i in range(len(data)-2, -1, -1):
                if z in data[i]['zs']:
                    c += 1
                else:
                    break
            if c >= 5: scores[z] += W['D6a']
            elif c >= 4: scores[z] += W['D6b']
            elif c >= 3: scores[z] += W['D6c']
    
    # D7
    r10 = Counter()
    for d in data[-10:]:
        r10.update(d['zs'])
    for z in zs:
        h = r10.get(z, 0)
        if h >= 5: scores[z] += W['D7a']
        elif h >= 4: scores[z] += W['D7b']
        elif h >= 3: scores[z] += W['D7c']
        elif h >= 2: scores[z] += W['D7d']
        else: scores[z] += W['D7e']
    
    # D8
    for z in zs:
        if miss.get(z, 0) > W['D8t']:
            scores[z] += W['D8']
    
    # D9
    for p in range(7):
        pc = Counter()
        for d in data[-20:]:
            pc[d['zs'][p]] += 1
        for z, c in pc.most_common(2):
            if c >= 3:
                scores[z] += 0.5
    
    # D10
    for i in range(len(last['nums'])-1):
        if last['nums'][i+1] - last['nums'][i] == 1:
            scores[last['zs'][i]] += 1
    
    # D11
    for z in last['zs']:
        scores[z] += W['D11']
    
    # D12
    cold = [z for z in zs if miss.get(z, 0) >= 5]
    hot = [z for z in zs if r10.get(z, 0) >= 4]
    for z in cold:
        if miss.get(z, 0) >= 2:
            scores[z] += W['D12a']
    for z in hot:
        scores[z] += W['D12b']
    
    # D13
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
            scores[z] += W['D13a']
        elif miss.get(z, 0) > ai:
            scores[z] += W['D13b']
    
    # D14
    pairs = [('鼠','龙'), ('龙','蛇'), ('蛇','马'), ('马','羊'), ('羊','猴'),
             ('猴','鸡'), ('鸡','狗'), ('狗','猪'), ('猪','牛'), ('牛','虎'),
             ('虎','兔'), ('兔','鼠')]
    for z1, z2 in pairs:
        if z1 in last['zs']: scores[z2] += 3
        if z2 in last['zs']: scores[z1] += 3
    
    # D15
    odd = sum(1 for n in last['nums'] if n % 2 == 1)
    if odd >= 5:
        for z in ['马','虎','龙','蛇','狗']:
            scores[z] += 5
    else:
        for z in ['鼠','牛','兔','羊','猴','鸡','猪']:
            scores[z] += 5
    
    # D16
    big = sum(1 for n in last['nums'] if n >= 25)
    if big >= 5:
        for z in ['马','蛇','龙','虎','牛']:
            scores[z] += 4
    else:
        for z in ['鼠','兔','羊','猴','鸡','狗','猪']:
            scores[z] += 4
    
    # D17
    s = sum(last['nums'])
    if 100 <= s <= 150:
        for z in zs:
            scores[z] += 2
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[0][0], ranked[0][1], scores  # 只返回预测TOP1

# 主程序
all_data = load(r'D:\Desktop\数据2026.xls', 7, 14)
total = len(all_data)

print("xac优化版 - D5加强")
print("="*50)
print("数据: {}期".format(total))

if total >= 51:
    # 预测最新一期
    pred, score, all_scores = predict(all_data)
    print()
    print("第{}期预测: {}".format(total+1, pred))
    print("得分: {:.1f}".format(score))
    
    # 回测
    print()
    print("="*50)
    print("回测结果（第51-86期，共{}期）".format(total-50))
    print("="*50)
    
    start = 50
    hits = 0
    for i in range(start, total):
        train = all_data[:i]
        actual = all_data[i]
        pred_i, _, _ = predict(train)
        actual_set = set(actual['zs'])
        hit = "✓" if pred_i in actual_set else "✗"
        if pred_i in actual_set:
            hits += 1
        print("{:>4} | {:^4} | {} | {}".format(i+1, pred_i, hit, ','.join(sorted(actual_set))))
    
    n = total - start
    print()
    print("="*50)
    print("TOP1命中率: {}/{} = {:.1f}%".format(hits, n, hits/n*100))
else:
    print("数据不足51期，无法预测")
