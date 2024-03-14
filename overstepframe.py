import threading
class FreshestFrame(threading.Thread):
        cnt = 1
        def __init__(self, capture, name='FreshestFrame'):
            self.capture = capture
            # assert self.capture.isOpened()

            self.cond = threading.Condition()
            self.running = False
            self.frame = None
            self.latestnum = 0

            self.callback = None

            super().__init__(name=name+f' {FreshestFrame.cnt}')
            FreshestFrame.cnt+=1
            self.start()
            
        def start(self):
            self.running = True
            super().start()

        def release(self, timeout=None):
            self.running = False
            self.join(timeout=timeout)
            self.capture.release()
            # self.__init__(self.capture)

        def run(self):
            counter = 0
            while self.running:
                (rv, img) = self.capture.read()
                # assert rv
                counter += 1

                with self.cond:
                    self.frame = img if rv else None
                    self.latestnum = counter
                    self.cond.notify_all()

                if self.callback:
                    self.callback(img)

        def read(self, wait=True, seqnumber=None, timeout=None):
            with self.cond:
                if wait:
                    if seqnumber is None:
                        seqnumber = self.latestnum+1
                    if seqnumber < 1:
                        seqnumber = 1
                    
                    rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                    if not rv:
                        return (self.latestnum, self.frame)

                return (self.latestnum, self.frame)