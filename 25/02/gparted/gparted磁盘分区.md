@[toc]
## 背景

最初给 linux 划分磁盘分区的时候，没有给/分区足够大的空间。我只给/分区分配了60G的空间，1年时间不到就全部用完了。我遇到过最严重的情况是：在/分区空间全部用完后，导致电脑无法正常开机（这个问题也有办法解决，但当时我看到电脑无法开机的时候，人确实慌了）。

## 简介

详细介绍如何使用 Gparted 工具，给linux主机重新划分磁盘空间

## 磁盘分区介绍

让GPT帮我规划一下磁盘分区，它推荐给 /分区 100G的空间。大家磁盘空间足够的话，一次性划分到位，这样就不用后续折腾了。

![image-20250207101420054](https://i-blog.csdnimg.cn/img_convert/e4dfea3cc10c7d52c9e5aef59c80979b.png)

如果你有特定用途，可以考虑**额外分区**：

- **`/var` (可选)**：如果要运行数据库、Web 服务器，建议单独分出 **50~100GB** 给 `/var`，避免日志过多导致 `/` 爆满。
- **`/tmp` (可选)**：如果有大文件临时处理需求（如视频渲染、编译），可以单独划 10GB 给 `/tmp`，避免临时文件撑爆 `/`。

如果不划分 /var 和 /tmp 分区，这些都会占用 / 分区的空间。尤其是 /var 分区占用空间会很大，因为Mysql数据库和Neo4j图数据库的默认存储地址在/var分区。在/分区的空间快用完时，我尝试过把数据库的存储地址更改到其他分区，比如：[如何在Ubuntu上更改MySQL数据存储路径](https://blog.csdn.net/sjxgghg/article/details/142875697)。但这些终究是杯水车薪，我60G /分区空间还是很快被用完了。于是重新划分磁盘空间的任务就提上了日程。



## 相关参考教程

**前提**：我尝试的分区扩容是在一块磁盘上进行，没有插入新的磁盘。/分区与/home分区是同一块物理磁盘，把/home分区多余的空间移给/分区。

这是我当时看的一个视频教程：【教你用Gparted无损调整Linux分区大小 . https://www.bilibili.com/video/BV1sz4y1R7QV/
他使用的是一个虚拟机，而我是实际的物理机器，故我没有严格按照他的这个流程进行。

> 大家在看完视频后，应该已经知道磁盘分区的原理了。
>
> 1. 构建空白分区，缩小一个大空间磁盘分区的容量，此时会多出一块未使用的空白分区；
> 2. 移动分区，把空白分区移动到待扩容的分区后；
> 3. 合并分区，合并空白分区到待扩容分区上；



我首先尝试了在linux系统上安装 gparted，尝试进行磁盘分区调整，但是无法调整。因为电脑在开机的状态，/ 和 /home 分区都是只读，不允许调整。那就还有一种方法，制作 gparted U盘启动器，电脑启动时从U盘Gparted启动，进行分区调整，然后再重启电脑进入linux系统。

下述是我在制作 gparted U盘启动器时，参考与用到的一些资料：

* https://blog.csdn.net/minen/article/details/50768895
* https://blog.csdn.net/qq_40682833/article/details/120318014
* https://gparted.org/download.php gparted ISO 下载地址
* https://www.ultraiso.com/ 



## 实际操作

下载 Gparted 的ISO镜像，再用ultraiso制作U盘启动器（虽然ultraiso是付费软件，免费试用就行）。

![image-20250207104148385](https://i-blog.csdnimg.cn/img_convert/4ba89c16e95a31e9a5d611cdd0b2ab87.png)

U盘启动器，制作完成后，其中的文件如下图所示：

![image-20250207104216293](https://i-blog.csdnimg.cn/img_convert/b32e580188823c5e5e881a5acc798ad5.png)

重启电脑，从U盘启动：

选择第一个默认设置

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6a27bb7a0d314a2587b41e0ebf94954d.png)


默认 Don’t touch keycap，选择 \<OK>:

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c11d52c452144e8bf19ba8f3406174f.png)


然后输入26，选择中文；

【提示】：大家在点击 调整大小/移动 后，不会立即执行，需要点击顶部的绿色✅才会执行



缩小大分区的容量，右键选择 resize，再点调整大小/移动，创建出空白分区：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d3cee1eee55c434db47d894d0ab7bc1d.png)


如下图所示，显示空白分区创建完成：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0865a91867dc430c8a17981f09461ce6.png)


移动分区，把空白分区移动到待扩容分区后：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/084ac9f5a44640a4b198314804e4b265.png)


使用鼠标拖拽移动空白分区的位置：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bdec1ca4e30a463fb664bdf00b9e666a.png)


会有下述警告⚠️，我们没有操作 /boot 分区，可以放心移动：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b88096987f0042c68943bff07a800e45.png)


如下图所示，空白分区已经移动到待扩容分区后，后续进行合并即可：

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2b5084078810499ca2aa47f679330689.png)



对待扩容分区，选择 resize，鼠标选择右侧滑块移动扩容：


![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6d42527aca96442a9439325a9909ff33.png)


至此，分区调整完成，/ 分区 容量从 61G 扩大为 161G。

重启电脑，正常进入linux即可！