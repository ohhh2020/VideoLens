"""只向调用者返回可以公开的错误，不透传密钥、模型输入和内部路径。"""


class VideoLensError(Exception):
    """业务错误：code 供程序判断，message 供使用者阅读。"""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class TaskInterrupted(Exception):
    """协作式停止信号；暂停和取消都要先释放资源，再更新任务终态。"""

    def __init__(self, action: str):
        self.action = action
        super().__init__(action)
