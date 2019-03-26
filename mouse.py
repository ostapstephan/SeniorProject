# from pymouse import PyMouseEvent

# def fibo():
    # a = 0
    # yield a
    # b = 1
    # yield b
    # while True:
        # a, b = b, a+b
        # yield b

# class Clickonacci(PyMouseEvent):
    # def __init__(self):
        # PyMouseEvent.__init__(self)
        # self.fibo = fibo()

    # def click(self, x, y, button, press):
        # '''Print Fibonacci numbers when the left click is pressed.'''
        # if button == 1:
            # if press:
                # print(self.fibo.next())
        # else:  # Exit if any other mouse button used
            # self.stop()

# C = Clickonacci()
# C.run()
import Xlib
import Xlib.display

def main():
    display = Xlib.display.Display('0')
    root = display.screen().root
    root.change_attributes(event_mask=
        Xlib.X.ButtonPressMask | Xlib.X.ButtonReleaseMask)

    while True:
        event = root.display.next_event()
        print(event)

if __name__ == "__main__":
    main()
