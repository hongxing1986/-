# 验证近期加权版规律库的命中率
import pandas as pd
from collections import Counter, defaultdict
import sys
sys.stdout.reconfigure(encoding='utf-8')

zm = {1:'马',13:'马',25:'马',37:'马',49:'马',2:'蛇',14:'蛇',26:'蛇',38:'蛇',3:'龙',15:'龙',27:'龙',39:'龙',4:'兔',16:'兔',28:'兔',40:'兔',5:'虎',17:'虎',29:'虎',41:'虎',6:'牛',18:'牛',30:'牛',42:'牛',7:'鼠',19:'鼠',31:'鼠',43:'鼠',8:'猪',20:'猪',32:'猪',44:'猪',9:'狗',21:'狗',33:'狗',45:'狗',10:'鸡',22:'鸡',34:'鸡',46:'鸡',11:'猴',23:'猴',35:'猴',47:'猴',12:'羊',24:'羊',36:'羊',48:'羊'}
zs_all = ['鼠','牛','虎','兔','龙','蛇','马','羊','猴','鸡','狗','猪']

def load(fp, sc=6, ec=13):
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

def calc_miss(train):
    last = train[-1]['zs']
    miss = {}
    for z in zs_all:
        if z in last:
            miss[z] = 0
        else:
            m = 1
            for i in range(len(train)-2, -1, -1):
                if z in train[i]['zs']:
                    break
                m += 1
            miss[z] = m
    return miss

def s1(train):
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    return c.most_common(1)[0][0]

def s2(train):
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    return c.most_common(1)[0][0]

def s3(train):
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    miss = calc_miss(train)
    for z,_ in c.most_common():
        if miss[z] >= 2:
            return z
    return c.most_common(1)[0][0]

def s4(train):
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    miss = calc_miss(train)
    for z,_ in c.most_common():
        if miss[z] >= 2:
            return z
    return c.most_common(1)[0][0]

def s5(train):
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z in last:
            return z
    return c.most_common(1)[0][0]

def s6(train):
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z in last:
            return z
    return c.most_common(1)[0][0]

def s7(train):
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-5:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z not in last:
            return z
    return c.most_common(1)[0][0]

def s8(train):
    last = set(train[-1]['zs'])
    c = Counter()
    for d in train[-10:]: c.update(d['zs'])
    for z,_ in c.most_common():
        if z not in last:
            return z
    return c.most_common(1)[0][0]

strategies = [
    ('S1', s1, '近5最热'),
    ('S2', s2, '近10最热'),
    ('S3', s3, '近5热+漏>=2'),
    ('S4', s4, '近10热+漏>=2'),
    ('S5', s5, '近5热+上期'),
    ('S6', s6, '近10热+上期'),
    ('S7', s7, '近5热+不在上'),
    ('S8', s8, '近10热+不在上'),
]
strategy_funcs = {name: func for name, func, _ in strategies}

def extract_features(train):
    c = Counter()
    for d in train: c.update(d['zs'])
    miss = calc_miss(train)
    
    repeat_count = 0
    for i in range(1, len(train)):
        last_zs = set(train[i-1]['zs'])
        curr_zs = set(train[i]['zs'])
        repeat_count += len(last_zs & curr_zs)
    avg_repeat = repeat_count / (len(train) - 1)
    
    top3_count = sum(cnt for z, cnt in c.most_common(3))
    total_count = sum(c.values())
    concentration = top3_count / total_count
    
    miss_std = sum((miss[z] - sum(miss.values())/12)**2 for z in zs_all) ** 0.5 / 12
    
    hot1 = c.most_common(1)[0][1]
    hot2 = c.most_common(2)[1][1] if len(c) > 1 else 0
    hot_ratio = hot1 / hot2 if hot2 > 0 else 3
    
    c5 = Counter()
    for d in train[-5:]: c5.update(d['zs'])
    hot5_top = c5.most_common(1)[0][1]
    
    last_zs = set(train[-1]['zs'])
    last_in_hot5 = sum(1 for z in last_zs if z in [x[0] for x in c5.most_common(3)])
    
    features = {
        'repeat': 1 if avg_repeat < 2.0 else (2 if avg_repeat < 2.5 else (3 if avg_repeat < 3.0 else 4)),
        'concentration': 1 if concentration < 0.35 else (2 if concentration < 0.40 else 3),
        'miss_volatility': 1 if miss_std < 0.3 else (2 if miss_std < 0.5 else 3),
        'hot_ratio': 1 if hot_ratio < 1.3 else (2 if hot_ratio < 1.5 else 3),
        'hot5_level': 1 if hot5_top < 5 else (2 if hot5_top < 7 else 3),
        'last_in_hot': 1 if last_in_hot5 >= 3 else (2 if last_in_hot5 >= 2 else 3),
    }
    
    return features

class PatternLibrary:
    def __init__(self, recent_weight=3.0):
        self.library = {}
        self.strategy_map = {}
        self.recent_weight = recent_weight
    
    def build(self, data):
        n = len(data)
        
        for window_start in range(0, n - 11):
            window_end = window_start + 10
            train = data[window_start:window_end]
            actual = set(data[window_end]['zs'])
            
            features = extract_features(train)
            feature_key = tuple(sorted(features.items()))
            
            time_weight = 1.0 + (window_start / (n - 20)) * (self.recent_weight - 1)
            
            for name, func in strategy_funcs.items():
                pred = func(train)
                hit = 1 if pred in actual else 0
                
                if feature_key not in self.library:
                    self.library[feature_key] = {
                        'samples': [],
                        'strategy_scores': defaultdict(float)
                    }
                
                self.library[feature_key]['samples'].append('{}-{}期'.format(window_start+1, window_end))
                self.library[feature_key]['strategy_scores'][name] += hit * time_weight
        
        for key, info in self.library.items():
            best = max(info['strategy_scores'].items(), key=lambda x: x[1])
            self.strategy_map[key] = {
                'strategy': best[0],
                'weighted_score': best[1],
                'count': len(info['samples'])
            }
    
    def predict(self, train):
        features = extract_features(train)
        feature_key = tuple(sorted(features.items()))
        
        if feature_key in self.strategy_map:
            return self.strategy_map[feature_key]['strategy'], '精确', feature_key
        else:
            best_similar = None
            best_diff = float('inf')
            for key in self.strategy_map:
                diff = sum(1 for a, b in zip(sorted(features.items()), key) if a[1] != b[1])
                if diff < best_diff:
                    best_diff = diff
                    best_similar = key
            return self.strategy_map[best_similar]['strategy'], '模糊({})'.format(best_diff), best_similar
    
    def get_prediction(self, train):
        strategy_name, match_type, key = self.predict(train)
        func = strategy_funcs[strategy_name]
        return func(train), strategy_name, match_type, key

# 主程序
data_file = r'D:\Desktop\数据2026_correct.xlsx'
data = load(data_file)

print('='*70)
print('规律库验证 - 近期加权版 (权重=3)')
print('='*70)
print()

# 构建规律库
library = PatternLibrary(recent_weight=3.0)
library.build(data)

# 测试近10期
print('近10期预测结果 (第79-88期):')
print('-'*70)
print('{:<8} | {:^10} | {:^10} | {:^10} | {}'.format('期数', '策略', '预测', '结果', '实际'))
print('-'*70)

recent_hits = 0
for i in range(78, 88):
    train = data[i-10:i]
    actual = set(data[i]['zs'])
    
    pred, strategy, match_type, key = library.get_prediction(train)
    hit = pred in actual
    if hit:
        recent_hits += 1
    
    print('{:<8} | {:^10} | {:^10} | {}{:<6} | {}'.format(
        '第{}期'.format(i+1),
        strategy,
        pred,
        '✓' if hit else '✗',
        match_type,
        ','.join(sorted(actual)[:4])
    ))

print()
print('='*70)
print('验证结果')
print('='*70)
print()
print('近10期命中率: {}/10 = {:.0f}%'.format(recent_hits, recent_hits/10*100))
print()

# 对比各策略
print('='*70)
print('各策略命中率对比')
print('='*70)
print()

for name, func, desc in strategies:
    hits = 0
    for i in range(78, 88):
        train = data[i-10:i]
        actual = set(data[i]['zs'])
        pred = func(train)
        if pred in actual:
            hits += 1
    print('{:<15}: {}/10 = {:.0f}%'.format(desc, hits, hits/10*100))
