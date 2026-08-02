jav-trans 安装与使用
====================

双击 jav-trans-setup.exe。

第一次运行会做两件事：

1. 测速并询问代理。
   程序会读取 PyTorch 官方源上的 torch 安装包，报出实测速度和「按这个速度
   要下多久」。速度可以接受就直接回车跳过；太慢或显示「连接失败」，就按
   y 填写本地代理（Clash / v2ray 等）的地址和端口。
   填写的代理会保存到 .env，之后下载 ASR 模型时也会自动使用。

2. 安装运行环境。
   控制台会实时显示 PyTorch 等依赖的下载进度，安装后约占 3.3GB 磁盘。
   中途断网或关掉窗口都不要紧，重新双击 jav-trans-setup.exe 会接着装，
   已经下好的部分不会重下。

装完会自动打开程序窗口。以后每次都双击同一个 jav-trans-setup.exe 启动，
它会跳过安装直接启动。

ASR 模型（约 3.9GB）在第一次转录时才下载，同样走上面设置的代理，
存放在本目录的 models\ 下。

需要注意
--------

* 请把整个文件夹解压到有至少 15GB 空闲空间的位置，不要放在 C:\Program Files
  或其他需要管理员权限的目录——程序要在自己的目录里写入 .venv、models、tmp。
* 需要 NVIDIA 显卡和较新的显卡驱动。驱动过旧时程序会在界面上提示更新。
* 如果窗口打不开，安装 Microsoft Edge WebView2 运行时：
  https://developer.microsoft.com/en-us/microsoft-edge/webview2/
* 出问题时，tmp\log\ 下的 .run.log 可以直接附在反馈里。

命令行参数（可选）
------------------

  jav-trans-setup.exe --proxy http://127.0.0.1:7890   指定代理，不询问
  jav-trans-setup.exe --yes                           直连安装，不询问
  jav-trans-setup.exe --reinstall                     重装运行环境
  jav-trans-setup.exe --install-only                  只安装，不启动
