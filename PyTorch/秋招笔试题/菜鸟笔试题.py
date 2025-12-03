import sys

n = int(sys.stdin.readline())

L = []
count = 0
is_sorted = True
for i in range(2 * n):
    input = sys.stdin.readline().split()
    action = input[0]
    if action == "push":
        value = int(input[1])
        if L and L[-1] < value:
            is_sorted = False
        L.append(value)
    else:
        if not is_sorted:
            L.sort()
            is_sorted = True
            count += 1
        L.pop()

print(count)