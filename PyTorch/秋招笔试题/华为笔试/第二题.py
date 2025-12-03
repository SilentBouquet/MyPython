import sys


def is_flower(num):
    if len(str(num)) != 3:
        return False
    L = list(str(num))

    sum = 0
    for i in L:
        item = int(i)
        sum += item * item * item
    if sum == num:
        return True
    else:
        return False


input = list(sys.stdin.readline().strip())

if len(input) < 2:
    print(0)
elif len(input) == 2:
    num = ord(input[0]) + ord(input[1])
    if is_flower(num):
        print(1)
    else:
        print(0)
else:
    cnt = 0
    is_true = False
    index = 0
    current_str = input
    first_flower = ""
    first_num = 0
    while True:
        if len(current_str) < 2:
            break

        num = ord(current_str[0]) + ord(current_str[1])
        index += 2

        while not is_flower(num) and index < len(input):
            num += ord(input[index])
            index += 1

        if not is_flower(num) and index >= len(input):
            break

        if not first_flower:
            first_flower = input[:index]
            first_num = num

        if index == len(input):
            is_true = True
            cnt += 1
            break

        current_str = input[index:]
        cnt += 1

    if not is_true:
        print(0)
    else:
        if cnt == 1:
            print(1)
        else:
            first_num += ord(input[len(first_flower)])
            index = len(first_flower) + 1
            while not is_flower(first_num) and index < len(input):
                first_num += ord(input[index])
                index += 1

            if not is_flower(first_num) and index >= len(input):
                print(cnt)
            else:
                is_second = False
                current_str = input[index:]
                while True:
                    if len(current_str) < 2:
                        break

                    num = ord(current_str[0]) + ord(current_str[1])
                    index += 2
                    while not is_flower(num) and index < len(input):
                        num += ord(input[index])
                        index += 1

                    if not is_flower(num) and index >= len(input):
                        break

                    if index == len(input):
                        is_second = True
                        break

                    current_str = input[index:]

                if is_second:
                    print(-1)
                else:
                    print(cnt)
