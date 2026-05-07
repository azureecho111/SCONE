#!/bin/bash

# 数据转移管理器

# 配置
PRODUCER_SCRIPT="scripts_14.py"
CONSUMER_SCRIPT="scripts_248.py"
LOG_DIR="./logs"

# 创建日志目录
mkdir -p $LOG_DIR

# 函数：启动生产者（在B服务器运行）
start_producer() {
    echo "启动生产者（B服务器）..."
    nohup python3 $PRODUCER_SCRIPT > $LOG_DIR/producer.log 2>&1 &
    echo $! > $LOG_DIR/producer.pid
    echo "生产者PID: $(cat $LOG_DIR/producer.pid)"
}

# 函数：启动消费者（在A服务器运行）
start_consumer() {
    echo "启动消费者（A服务器）..."
    nohup python3 $CONSUMER_SCRIPT > $LOG_DIR/consumer.log 2>&1 &
    echo $! > $LOG_DIR/consumer.pid
    echo "消费者PID: $(cat $LOG_DIR/consumer.pid)"
}

# 函数：停止进程
stop_process() {
    local process=$1
    local pid_file="$LOG_DIR/${process}.pid"

    if [ -f $pid_file ]; then
        local pid=$(cat $pid_file)
        echo "停止$process (PID: $pid)..."
        kill $pid
        rm $pid_file
        echo "$process 已停止"
    else
        echo "$process 未运行"
    fi
}

# 函数：查看状态
status() {
    echo "=== 进程状态 ==="
    for process in producer consumer; do
        local pid_file="$LOG_DIR/${process}.pid"
        if [ -f $pid_file ]; then
            local pid=$(cat $pid_file)
            if ps -p $pid > /dev/null; then
                echo "$process: 运行中 (PID: $pid)"
            else
                echo "$process: PID文件存在但进程未运行"
                rm $pid_file
            fi
        else
            echo "$process: 未运行"
        fi
    done

    echo -e "\n=== 最近日志 ==="
    tail -n 5 $LOG_DIR/*.log 2>/dev/null
}

# 主逻辑
case "$1" in
    "start-producer")
        start_producer
        ;;
    "start-consumer")
        start_consumer
        ;;
    "start-all")
        start_producer
        sleep 2
        start_consumer
        ;;
    "stop-producer")
        stop_process "producer"
        ;;
    "stop-consumer")
        stop_process "consumer"
        ;;
    "stop-all")
        stop_process "producer"
        stop_process "consumer"
        ;;
    "status")
        status
        ;;
    *)
        echo "用法: $0 {start-producer|start-consumer|start-all|stop-producer|stop-consumer|stop-all|status}"
        exit 1
        ;;
esac