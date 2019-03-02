import saleae

if __name__ == "__main__":
    try:
        s = saleae.Saleae()
        time.sleep(1)
        print s.get_active_device()
    except:
        print "logic software not open or device not found"
