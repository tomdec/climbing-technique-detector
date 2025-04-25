from cv2 import imshow, imread, waitKey, destroyAllWindows

if __name__ == '__main__':
    path = "./data/img/techniques/test/INSIDE_FLAG/How to Flag - A Climbing Technique for Achieving Balance__11306__31.png"
    imshow(path, imread(path))

    while True:
        res = waitKey(0)
        print('You pressed %d (0x%x), LSB: %d (%s)' % (res, res, res % 256,
            repr(chr(res%256)) if res%256 < 128 else '?'))
        
        if res & 0xFF == ord('q'):
            print('quitting')
            break

    destroyAllWindows()