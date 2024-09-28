import random

count = 5000
count1 = 0
count2 = 0
Arr = ["没中", "没中", "没中"]
num = random.randint(0, 2)
Arr[num] = "中"
for i in range(count):
    choice1 = random.randint(0, 2)
    choice2 = random.randint(0, 2)
    if choice1 == choice2:
        continue
    else:
        count1 += 1
        if Arr[choice1] == "中":
            count2 += 1

print(count1, count2)
print(count2 / count1)