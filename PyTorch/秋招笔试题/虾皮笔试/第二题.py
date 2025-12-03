import sys

input = [i.strip('“').strip('”') for i in sys.stdin.readline().strip().split(',  ')]

v1 = input[0].split('.')
v2 = input[1].split('.')

is_same = True
pos1 = pos2 = 0
while pos1 < len(v1) or pos2 < len(v2):
    x = int(v1[pos1]) if pos1 < len(v1) else 0
    y = int(v2[pos2]) if pos2 < len(v2) else 0
    if x > y:
        is_same = False
        print(1)
        break
    elif x < y:
        is_same = False
        print(-1)
        break
    else:
        pos1 += 1
        pos2 += 1

if is_same:
    print(0)