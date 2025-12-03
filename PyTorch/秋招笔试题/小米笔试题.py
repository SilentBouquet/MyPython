import sys

t = int(sys.stdin.readline().strip())
for j in range(t):
    input = sys.stdin.readline().strip().split()
    n = int(input[0])
    k = int(input[1])
    input = sys.stdin.readline().strip().split()
    W = [int(i) for i in input]
    right = int(k / 2)
    left = k - right
    count = 0
    start = 0
    end = n - 1
    while start <= end:
        if right <= 0 and left <= 0:
            break
        if left >= W[start] > 0:
            left -= W[start]
            W[start] = 0
            start += 1
            count += 1
        else:
            W[start] -= left
            left = 0
        if right >= W[end] > 0:
            right -= W[end]
            W[end] = 0
            end -= 1
            count += 1
        else:
            W[end] -= right
            right = 0
    print(count)