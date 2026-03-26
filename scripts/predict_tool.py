"""
生肖预测工具 - 11维度综合分析法
用于：根据历史数据预测下一期最可能出现的生肖

使用方法：
1. 把新一期的7个数字发给小元
2. 小元会自动运行此脚本
3. 输出预测结果

预测维度：
1. 各位置数字频率
2. 各位置生肖热度
3. 各位置大小分布
4. 各位置单双分布
5. 近10期趋势加权
6. 历史遗漏
7. 当前遗漏
8. 历史连开
9. 当前连开
10. 相邻期规律
11. 间隔规律
"""

import pandas as pd
from collections import Counter
import sys
import os

# ==================== 常量定义 ====================
ZODIAC_MAP = {
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

ZODIACS = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']

def size(n):
    """大小判断：1-24小，25-49大"""
    return '小' if n <= 24 else '大'

def parity(n):
    """单双判断：奇数单，偶数双"""
    return '单' if n % 2 == 1 else '双'

def load_data(file_path):
    """从Excel加载数据"""
    df = pd.read_excel(file_path, header=None)
    data = []
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.notna(row[5]):
            period = str(row[5]).replace('期', '').strip()
            nums = []
            for col in range(6, 13):
                val = row[col]
                if pd.notna(val):
                    s = str(val).strip()
                    if s.isdigit():
                        n = int(s)
                        if 1 <= n <= 49:
                            nums.append(n)
            if len(nums) == 7:
                data.append({
                    'period': int(period),
                    'numbers': nums,
                    'zodiacs': [ZODIAC_MAP[n] for n in nums]
                })
    return data

def predict_zodiac(data):
    """11维度综合预测"""
    scores = {z: 0 for z in ZODIACS}
    
    # 1. 各位置生肖热度
    pos_z_freq = {pos: Counter() for pos in range(7)}
    for d in data:
        for pos in range(7):
            pos_z_freq[pos][d['zodiacs'][pos]] += 1
    
    for pos in range(7):
        for z, cnt in pos_z_freq[pos].most_common(5):
            scores[z] += cnt * (7 - pos) * 0.5
    
    # 2. 近10期趋势
    recent10 = data[-10:] if len(data) >= 10 else data
    recent_z = Counter()
    for d in recent10:
        recent_z.update(d['zodiacs'])
    max_recent = max(recent_z.values()) if recent_z else 1
    for z in ZODIACS:
        scores[z] += (recent_z.get(z, 0) / max_recent) * 10
    
    # 3. 历史最大遗漏
    max_missing = {z: 0 for z in ZODIACS}
    streak = {z: 0 for z in ZODIACS}
    for d in data:
        for z in ZODIACS:
            if z in d['zodiacs']:
                if streak[z] > max_missing[z]:
                    max_missing[z] = streak[z]
                streak[z] = 0
            else:
                streak[z] += 1
    for z in ZODIACS:
        if streak[z] > max_missing[z]:
            max_missing[z] = streak[z]
    
    # 4. 当前遗漏
    current_miss = {z: 0 for z in ZODIACS}
    for z in ZODIACS:
        count = 0
        for i in range(len(data)-1, -1, -1):
            if z in data[i]['zodiacs']:
                break
            count += 1
        current_miss[z] = count
    
    # 遗漏加分
    for z in ZODIACS:
        if max_missing[z] > 0:
            ratio = current_miss[z] / max_missing[z]
            if ratio >= 0.8:
                scores[z] += 10
            elif ratio >= 0.6:
                scores[z] += 7
            elif ratio >= 0.4:
                scores[z] += 4
    
    # 5. 历史最长连开
    max_streak = {z: 0 for z in ZODIACS}
    consec = {z: 0 for z in ZODIACS}
    for d in data:
        for z in ZODIACS:
            if z in d['zodiacs']:
                consec[z] += 1
                if consec[z] > max_streak[z]:
                    max_streak[z] = consec[z]
            else:
                consec[z] = 0
    
    # 6. 当前连开（减分）
    current_consec = {z: 0 for z in ZODIACS}
    for z in ZODIACS:
        count = 0
        for i in range(len(data)-1, -1, -1):
            if z in data[i]['zodiacs']:
                count += 1
            else:
                break
        current_consec[z] = count
    
    for z in ZODIACS:
        if max_streak[z] > 0:
            ratio = current_consec[z] / max_streak[z]
            if ratio >= 1.0:
                scores[z] -= 8
            elif ratio >= 0.8:
                scores[z] -= 5
            elif ratio >= 0.5:
                scores[z] -= 2
    
    # 7. 相邻期规律
    last_zodiacs = set(data[-1]['zodiacs'])
    for z in ZODIACS:
        if z in last_zodiacs:
            scores[z] += 3
    
    # 8. 间隔规律
    intervals = {z: [] for z in ZODIACS}
    last_appear = {z: -1 for z in ZODIACS}
    for i, d in enumerate(data):
        for z in d['zodiacs']:
            if last_appear[z] >= 0:
                intervals[z].append(i - last_appear[z])
            last_appear[z] = i
    
    avg_intervals = {}
    for z in ZODIACS:
        if intervals[z]:
            avg_intervals[z] = sum(intervals[z]) / len(intervals[z])
        else:
            avg_intervals[z] = 999
    
    for z in ZODIACS:
        if avg_intervals[z] < 999:
            if current_miss[z] >= avg_intervals[z]:
                scores[z] += 5
            elif current_miss[z] >= avg_intervals[z] * 0.7:
                scores[z] += 3
    
    # 9. 各位置大小分布
    for pos in range(7):
        pos_sizes = Counter(size(d['numbers'][pos]) for d in data)
        hot_size = pos_sizes.most_common(1)[0][0]
        pos_size_recent = Counter(size(d['numbers'][pos]) for d in data[-5:])
        recent_hot = pos_size_recent.most_common(1)[0][0]
        if hot_size == recent_hot:
            for z in ZODIACS:
                if z in pos_z_freq[pos]:
                    scores[z] += 0.5
    
    # 10. 各位置单双分布
    for pos in range(7):
        pos_parity = Counter(parity(d['numbers'][pos]) for d in data)
        hot_parity = pos_parity.most_common(1)[0][0]
        pos_parity_recent = Counter(parity(d['numbers'][pos]) for d in data[-5:])
        recent_hot = pos_parity_recent.most_common(1)[0][0]
        if hot_parity == recent_hot:
            for z in ZODIACS:
                if z in pos_z_freq[pos]:
                    scores[z] += 0.5
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def predict_with_numbers(data):
    """预测具体数字"""
    if len(data) < 1:
        return []
    
    # 获取预测的生肖
    zodiac_scores = predict_zodiac(data)
    top_zodiacs = [z for z, s in zodiac_scores[:5]]
    
    # 获取当前遗漏最多的生肖（可能回补）
    current_miss = {}
    for z in ZODIACS:
        count = 0
        for i in range(len(data)-1, -1, -1):
            if z in data[i]['zodiacs']:
                break
            count += 1
        current_miss[z] = count
    
    # 获取位置热号数字
    pos_num_freq = {pos: Counter() for pos in range(7)}
    for d in data:
        for pos in range(7):
            pos_num_freq[pos][d['numbers'][pos]] += 1
    
    # 综合预测
    prediction = []
    used_nums = set()
    
    # 先按生肖预测
    for z in top_zodiacs:
        # 该生肖对应的数字
        zodiac_nums = [n for n, zz in ZODIAC_MAP.items() if zz == z]
        # 在历史中出现位置最热的数字
        for pos in range(7):
            for num in zodiac_nums:
                if num not in used_nums:
                    # 检查这个数字在所有位置的出现频率
                    total_freq = sum(pos_num_freq[p].get(num, 0) for p in range(7))
                    if total_freq > 0:
                        prediction.append(num)
                        used_nums.add(num)
                        break
            if len(prediction) >= 3:
                break
        if len(prediction) >= 3:
            break
    
    # 补充到7个
    recent_nums = []
    for d in data[-3:]:
        recent_nums.extend(d['numbers'])
    recent_freq = Counter(recent_nums)
    
    for num in range(1, 50):
        if num not in used_nums and num not in prediction:
            # 优先选近期热号
            if recent_freq[num] > 0:
                prediction.append(num)
                used_nums.add(num)
            if len(prediction) >= 7:
                break
    
    return prediction[:7]

def run_prediction(new_data):
    """运行预测主函数"""
    print("="*70)
    print("11维度综合分析 - 生肖预测")
    print("="*70)
    
    # 预测生肖
    zodiac_scores = predict_zodiac(new_data)
    
    print("\n【生肖评分排名】")
    for rank, (z, s) in enumerate(zodiac_scores, 1):
        bar = "█" * int(s/2)
        print(f"{rank:2d}. {z}: {s:6.1f}分 {bar}")
    
    # 预测数字
    numbers = predict_with_numbers(new_data)
    
    print(f"\n【预测结果】")
    print(f"推荐生肖: {[z for z,s in zodiac_scores[:3]]}")
    print(f"推荐数字: {numbers}")
    
    return zodiac_scores, numbers

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 默认从文件加载历史数据
    file_path = r"D:/Desktop/数据2026.xls"
    
    if os.path.exists(file_path):
        data = load_data(file_path)
        if len(data) >= 10:
            run_prediction(data)
        else:
            print("数据不足10期，无法预测")
    else:
        print(f"文件不存在: {file_path}")
        print("请提供数据文件路径")