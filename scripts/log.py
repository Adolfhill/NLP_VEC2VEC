import logging


# init and get a logger
def getLogger(filePath, index):
    # set log
    logger = logging.getLogger(name='logs{}'.format(index))  # 不加名称设置root logger
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    # use FileHandler to output to file
    fh = logging.FileHandler(filename=filePath, encoding="utf-8", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # use StreamHandler to output to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add handler
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
