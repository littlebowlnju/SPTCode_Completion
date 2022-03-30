import threading


class CodeCompletionTask:
    def __init__(self, code):
        self.code = code
        self.result = {}
        self.lock = threading.Condition()

    def get_result(self):
        # print("get result")
        with self.lock:
            self.lock.wait()
            # print("getting result waiting!!!")
        return self.result

    def set_result(self, result_dic):
        # print("set result")
        with self.lock:
            self.result = result_dic
            self.lock.notify()
