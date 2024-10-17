file1 = open("发货记录.csv", "r", encoding="utf-8")
file3 = open("中间表2.csv", "r", encoding="utf-8")
lines = file3.readlines()
i = 0

while True:
    if i >= len(lines):
        break
    line = lines[i]
    content1 = file1.readline().strip("\n").split(",")
    if content1[0] == "id":
        i += 1
        continue
    with open("客户下单信息.csv", "r", encoding="utf-8") as file2:
        while True:
            content2 = file2.readline().strip("\n").split(",")
            if content2[0] == "id":
                continue
            if eval(content2[0]) > eval(content1[4]) or content2[0] == "":
                new_line = line.strip() + "," + "null" + "\n"
                lines[i] = new_line
                break
            if content2[0] == content1[4]:
                new_line = line.strip() + "," + content2[1] + "\n"
                lines[i] = new_line
                break
    print(i)
    print(lines[i])
    i += 1

file3.close()
file4 = open("中间表3.csv", "w", encoding="utf-8")
file4.writelines(lines)
file4.close()