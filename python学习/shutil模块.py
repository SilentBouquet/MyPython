# shutil主要封装了文件和文件夹的相关操作，比如复制、粘贴、移动等
import shutil

'''
# 把dir/test.txt移动到dir2文件夹内
shutil.move("../操作文件/dir1/test.txt", "../操作文件/dir2")

# 复制两个文件句柄
f1 = open("../操作文件/dir2/test.txt", mode="rb")
f2 = open("../操作文件/dir1/test.txt", mode="wb")
shutil.copyfileobj(f1, f2)

# 执行两个文件路径，进行文件的复制
shutil.copyfile("../操作文件/dir1/test.txt", "../操作文件/dir1/test2.txt")

# 复制文件内容，文件权限也一起进行复制
shutil.copy("../操作文件/dir1/test.txt", "../操作文件/dir1/test3.txt")

# 复制文件内容、文件权限和修改时间
shutil.copy2("../操作文件/dir1/test.txt", "../操作文件/dir1/test4.txt")

# 修改权限和时间，不复制内容
shutil.copystat("../操作文件/dir1/test4.txt", "../操作文件/dir1/test.txt")

# 只拷贝权限
shutil.copymode("../操作文件/dir1/test3.txt", "../操作文件/dir1/test4.txt")

# 复制文件夹
shutil.copytree("../操作文件/dir1", "../操作文件/dir3")
'''

# 删除文件夹
shutil.rmtree("../操作文件/dir1")
shutil.rmtree("../操作文件/dir2")
shutil.rmtree("../操作文件/dir3")