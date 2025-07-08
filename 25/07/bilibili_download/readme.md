使用IDM插件从BiliBili下载视频，会得到两个m4s的文件，其中一个是音频，一个视频。

于是编写了一个python程序，利用ffmpeg工具，将两个m4s文件合并为一个mp4文件。
ffmpeg 可以使用cuda加速，我的显卡占用率可以跑到100%。



## ffmpeg 安装

下载地址：

- https://ffmpeg.org/download.html
- https://www.gyan.dev/ffmpeg/builds/

选择 ffmpeg-git-full.7z 完整版文件，解压之后，把该文件的bin目录添加到系统环境目录下就可以使用了

