import sys

input = sys.stdin.readline().strip().replace(' ', '').lower()
print("input为：", input)

s = ""
for i in range(len(input)):
    k = input[i]
    if 'a' <= k <= 'z':
        s += k

print("s为：", s)
start = 0
end = len(s) - 1
is_true = True
while True:
    print(f"此时：start为{start}，end为{end}")
    if end < start:
        break
    if s[end] != s[start]:
        is_true = False
        break
    start += 1
    end -= 1

if is_true:
    print('true')
else:
    print('false')