先连zju rvpn

在自己的terminal里面输入server:
ssh haoyang@10.105.100.212

password:
oooppp000

cd你自己建的文件夹

激活虚拟环境，我的虚拟环境里有pytorch，应该够用了
conda activate ER

开一个jupyter notebook端口 2222是我随便选的你可以换
jupyter notebook --ip=0.0.0.0 --no-browser --port=2222

在本地的terminal或者什么地方开个端口8888（随便挑的可换）连上服务器那边的端口2222
ssh -N -f -L localhost:8888:localhost:2222 haoyang@10.105.100.212

随便开个浏览器输入localhost:8888进入jupyter notebook
jupyter notebook 密码 oooppp000
进去以后就是正常的可视化的jupyter notebook
可以直接在那个界面上传文件下载文件用notebook测试

要跑训练的时候开个tmux的session
在里面跑防止中途退出训练被打断
下面是常用命令和快捷键
|      shortcut key & command            |      function                                |
|:--------------------------------------:|:--------------------------------------------:|
| tmux ls                                | see the tmux info                            |
| tmux new -s <session-name>             | create session                               |
| tmux a -t <session-name>               | attach to a session                          |
| tmux kill-session -t <session-name>    | kill a session                               |
| Ctrl + b + [                           | to see history (q to exit)                   |
| Ctrl + b + d                           | detach                                       |
| Ctrl + d                               | directly exit and kill current session       |
| Ctrl + b + s                           | list the sessions                            |


