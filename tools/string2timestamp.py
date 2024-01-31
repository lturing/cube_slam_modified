import datetime


def convertStringToTimestamp(src, des):
    with open(src, 'r', encoding='utf-8') as f:
        with open(des, 'w', encoding='utf-8') as rs: 
            t1 = None
            for line in f:
                line = line.strip()
                # 2011-10-03 14:57:27.032684544
                line = line.split(' ')
                year, month, day = map(int, line[0].split('-'))
                hour, minute = map(int, line[1].split(':')[:2])
                second = float(line[1].split(':')[-1])
                us = int((second - int(second)) * 1e6 + 0.5) 
                #print(' '.join(line), year, month, day, hour, minute, second, int(second), us)
                second = int(second)
                if t1 is None:
                    t1 = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=us)
                    rs.write('0.0' + '\n')
                else:
                    t2 = datetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=us)
                    delta = (t2 - t1).total_seconds()
                    rs.write(str(delta) + '\n')


if __name__ == '__main__':
    src = '/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/timestamps.txt'
    des = '/home/spurs/dataset/kitti_raw/2011_10_03/2011_10_03_drive_0047_sync/image_02/times.txt'
    convertStringToTimestamp(src, des)
