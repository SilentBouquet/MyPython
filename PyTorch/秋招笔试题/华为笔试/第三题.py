from collections import deque
import sys

input = [int(i) for i in list(sys.stdin.readline().strip().replace(' ', ''))]
print(input)
start = int(input[0])
end = int(input[2])
num = int(input[1])

dct = {}
for i in range(1, num+1):
    dct[i] = []

for i in range(num):
    input = [int(i) for i in list(sys.stdin.readline().strip().replace(' ', ''))]
    node = input[0]
    num_e = input[1]

    start = 2
    for i in range(num_e):
        dct[node].append([input[start], input[start+1]])
        start += 2

path = []
for i in range(num):
    path.append([])

print(dct)

d = deque([])
d.append(start)
visited = []
is_find = False
while len(d) > 0:
    root = d.popleft()
    print("弹出：", root)
    if root != end and len(dct[root]) > 0:
        for node in dct[root]:
            if node[0] not in visited:
                d.append(node[0])
                path[root-1].append(node[0])
                print(f"从{root}可以到：", node[0])
        visited.append(root)
    else:
        is_find = True
        break

if is_find:
    print(path)
else:
    print(0)