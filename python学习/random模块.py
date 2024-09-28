import random

print(random.random())  # (0, 1)

print(random.uniform(2, 3))  # 随机小数

print(random.randint(3, 8))  # 随机整数，能取到边界

lst = ["张无忌", "周杰伦", "潘玮柏", "周燕"]
print(random.choice(lst))  # 随机选择一个

lst = ["屠龙刀", "倚天剑", "金币", "青龙偃月刀", "经验书"]
# 每次随机爆出两个装备
print(random.sample(lst, 2))


# 练习：随机生成四位验证码
def rand_num():
    return random.randint(0, 9)


def rand_upper():
    return chr(random.randint(ord('A'), ord('Z')))


def rand_lower():
    return chr(random.randint(ord('a'), ord('z')))


def rand_code(n):
    lst = []
    for i in range(n):
        which = random.randint(1, 3)
        if which == 1:
            s = str(rand_num())
        elif which == 2:
            s = rand_upper()
        else:
            s = rand_lower()
        lst.append(s)
    return "".join(lst)


print(rand_code(6))