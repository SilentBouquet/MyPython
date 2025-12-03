import sys

input = sys.stdin.readline().strip()
L = list(input)


is_val = True
if len(L) != 9:
    is_val = False
    print("Invalid")

s = L[:2]
num = L[2:8]
r = L[-1]

sum = 0
if is_val:
    for item in s:
        if 'A' <= item <= 'Z':
            sum += ord(item)
        else:
            is_val = False
            print('Invalid')
            break

if is_val:
    for item in num:
        if '0' <= item <= '9':
            sum += ord(item)
        else:
            is_val = False
            print('Invalid')
            break

if is_val:
    result = sum % 26 + ord('A')
    if chr(result) != r:
        print(input[:8] + chr(result))
    else:
        print(input)
