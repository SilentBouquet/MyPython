with open("中间表4.csv", "r", encoding="utf-8") as file2:
    lines = file2.readlines()
file1 = open("中间表4.csv", "r", encoding="utf-8")
i = 0

while True:
    if i >= len(lines):
        break
    line = lines[i]
    content = file1.readline().strip("\n").split(",")
    if content[0] == "id":
        lines[i] = "id,create_time,sales_name,ywlx,khmc,hplx,cpgg,cppp,jz,dzdw,dj,hk\n"
        i += 1
        continue
    else:
        if content[2] == "null" or eval(content[8]) < 0:
            lines[i] = ""
    print(i)
    print(lines[i])
    i += 1

file1.close()
file3 = open("业务表.csv", "w", encoding="utf-8")
file3.writelines(lines)
file3.close()