#kill所有进程
bash vllm/kill_vllm.sh

# 使用前确保已经通过运行 start_cluster.sh 启动vllm集群，可以看看vllm文件夹下的log，只要gpu_7的log出现下面的内容，就证明8卡启动都完成了
bash vllm/start_cluster.sh

[1;36m(APIServer pid=2505478)[0;0m INFO:     Started server process [2505478]
[1;36m(APIServer pid=2505478)[0;0m INFO:     Waiting for application startup.
[1;36m(APIServer pid=2505478)[0;0m INFO:     Application startup complete.

# 然后运行：
# 在项目根目录下运行
bash Nginx/nginx_m.sh start

此时整个多卡vllm部署完成，直接访问 8668端口调用API

API_URL = "http://localhost:8668/v1"
API_KEY = "sk-12345678" # 这个是在vllm启动的时候固定写的，所以调用的时候要填上
MODEL = "Qwen3-8B-5"