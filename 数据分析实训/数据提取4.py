with open("中间表3.csv", "r", encoding="utf-8") as file3:
    lines = file3.readlines()
file1 = open("中间表3.csv", "r", encoding="utf-8")
i = 0

while True:
    if i >= len(lines):
        break
    line = lines[i]
    content1 = file1.readline().strip("\n").split(",")
    if content1[0] == "id":
        i += 1
        continue
    with open("客户信息.csv", "r", encoding="utf-8") as file2:
        while True:
            content2 = file2.readline().strip("\n").split(",")
            if content2[0] == "id":
                continue
            if content1[4] == "null" or eval(content2[0]) > eval(content1[4]):
                break
            if content2[0] == content1[4]:
                content = line.strip().split(",")
                new_line = content[0] + "," + content[1] + "," + content[2] + \
                    "," + content[3] + "," + content2[1] + "\n"
                lines[i] = new_line
                break
    print(i)
    print(lines[i])
    i += 1

file1.close()
file4 = open("中间表4.csv", "w", encoding="utf-8")
file4.writelines(lines)
file4.close()