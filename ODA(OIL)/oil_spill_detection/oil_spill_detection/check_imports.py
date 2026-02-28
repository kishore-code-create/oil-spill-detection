modules = ['flask','pymysql','torch','cv2','scipy','sklearn','roboflow','flask_mysqldb','pandas']
for m in modules:
    try:
        __import__(m)
        print(m + ' OK')
    except Exception as e:
        print(m + ' ERROR: ' + str(e))
