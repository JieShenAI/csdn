# shell 脚本

## IO

* 遍历文件夹

  ```shell
  
  ```

  

## 压缩

压缩文件：

```
zip xx/../xxx.zip path/xxx.txt
```

压缩文件夹：

```
zip -r xx/../xxx.zip path/xxx.txt
```

zip 的第一个参数是待生成压缩包，第二个参数是原始文件。默认会把原始文件的路径上的文件夹也压缩进去。
为了控制在解压时得到的路径，可使用 `cd path` 调整文件夹。第一个生成的压缩包的路径不会影响压缩包内的文件路径。

* `-j` 只在压缩包中保留纯文件，不保留路径。

