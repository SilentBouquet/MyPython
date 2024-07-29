file1 = open("发货记录.csv", "r", encoding="utf-8")
file2 = open("中间表.csv", "w", encoding="utf-8")

while True:
    content1 = file1.readline().strip("\n").split(",")
    if content1[0] == "":
        break
    file2.write(content1[0] + "," + content1[14] + "\n")

file1.close()
file2.close()