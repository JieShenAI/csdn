stata 文件与csv文件的读取测试

针对文件夹下所有的zip文件进行解压：
nohup find 分年数据/ -type f -exec unzip {} -d unzip_分年/  \; > unzip.log 2>&1 &