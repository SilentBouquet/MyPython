import sys
from collections import deque

n = int(sys.stdin.readline().strip())
inital = [int(i) for i in sys.stdin.readline().strip().split(' ')]
target = [int(i) for i in sys.stdin.readline().strip().split(' ')]

print("n为：", n)
print("inital为：", inital)
print("target为：", target)

dct = {}
for i, item in enumerate(inital):
    if item == 0:
        continue
    dct[i] = []
    d = deque([i])
    while len(d) > 0:
        j = d.popleft()
        if j * 2 + 1 < n and inital[j * 2 + 1] != 0:
            d.append(j * 2 + 1)
            dct[i].append(j * 2 + 1)
        if j * 2 + 2 < n and inital[j * 2 + 2] != 0:
            d.append(j * 2 + 2)
            dct[i].append(j * 2 + 2)

print("dct为：", dct)

cnt = 0
start = 0
while start < n:
    root = inital[start]
    if root == 0:
        start = start + 1
        continue
    e = target[start]
    if root != e:
        dist = (e - root) if e > root else (e + 5 - root)
        inital[start] = inital[start] + dist
        if inital[start] > 5:
            inital[start] -= 5
        cnt += dist
        if len(dct[start]) > 0:
            for i in dct[start]:
                inital[i] = inital[i] + dist
                if inital[i] > 5:
                    inital[i] -= 5
    start += 1

print(cnt)
print(inital)