import os

class Logger:
    def __init__(self, log_subdir="Naive"):
        self.log_dir = os.getenv("LOG_DIR", ".")
        self.log_path = os.path.join(self.log_dir, f"{log_subdir}.log")
        os.makedirs(self.log_dir, exist_ok=True)

        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def log(self, *args, **kwargs):
        """像 print 一样，把内容写入对应 step 的日志文件。"""
        if self.log_file is None:
            raise RuntimeError("请先调用 set_step(step) 指定日志 step。")

        message = " ".join(str(a) for a in args)
        self.log_file.write(message + "\n")
        self.log_file.flush()                   # 立刻写入磁盘，避免丢日志

    def close(self):
        if self.log_file:
            self.log_file.close()
            self.log_file = None
