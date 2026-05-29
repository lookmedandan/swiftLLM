const ws = new WebSocket('ws://localhost:8080/ws/chat');

ws.onopen = () => {
    ws.send("你好");
};

ws.onmessage = (event) => {
    if (event.data === "[DONE]") {
        console.log("对话结束");
    } else {
        console.log("收到:", event.data);
        // 实时显示到页面
        document.getElementById('chat-box').innerHTML += event.data;
    }
};