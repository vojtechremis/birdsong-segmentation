import time


class get_logger:
    def __init__(self, log_level=0, debug=False):
        self.log_level = log_level

    def getTime(self) -> str:
        current_time = time.localtime()
        return time.strftime('%H:%M:%S', current_time)

    def error(self, message):
        if self.log_level < 3:
            print(self.getTime() + '\t__ERROR__: ' + message)

    def warning(self, message):
        if self.log_level < 2:
            print(self.getTime() + '\t__WARNING:__ ' + message)

    def info(self, message):
        if self.log_level < 1:
            print(self.getTime() + '\t' + message)

    def debug(self, message):
        print(self.getTime() + '\t' + message)