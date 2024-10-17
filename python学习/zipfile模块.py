import zipfile

# 创建压缩包
f = zipfile.ZipFile('../操作文件/zip_dir/test.zip', 'w')
f.write("../操作文件/l1.txt")
f.write("../操作文件/l2.txt")
f.close()

# 解压缩
f = zipfile.ZipFile('../操作文件/zip_dir/test.zip', 'r')
# 直接全部解压缩
# f.extractall("../操作文件/zip_dir/test")
# 一个一个的解压缩
print(f.namelist())
for name in f.namelist():
    f.extract(name, "../操作文件/zip_dir/test")