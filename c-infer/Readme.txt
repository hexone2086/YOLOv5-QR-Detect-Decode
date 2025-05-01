
## 使用 CSI 摄像头
`main_csi.cc` 是使用 CSI IMX219 摄像头的测试用例，使用 gstreamer 和管道传输图像，所需依赖可通过包管理器安装。默认使用的 opencv-mobile 二进制版本支持 gstreamer + pipe 的方式读取。

```shell
sudo apt-get update
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt-get install gstreamer1.0-plugins-bad gstreamer1.0-plugins-good
```
