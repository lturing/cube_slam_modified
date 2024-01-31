from gps_to_xyz import load_calib


if __name__ == "__main__":
    calib_imu_to_velo_filepath =  '/home/spurs/dataset/kitti_raw/2011_10_03/calib_imu_to_velo.txt'
    velo_to_cam_filepath = '/home/spurs/dataset/kitti_raw/2011_10_03/calib_velo_to_cam.txt'
    cam_to_cam_filepath = '/home/spurs/dataset/kitti_raw/2011_10_03/calib_cam_to_cam.txt'

    calib = load_calib(calib_imu_to_velo_filepath, velo_to_cam_filepath, cam_to_cam_filepath)
    fx_0 = calib['K_cam0'][0][0]
    fx_2 = calib['K_cam1'][0][0]
    print(f"from cam0 to cam1(gray) baseline: {calib['b_gray']} meters, bf: {round(calib['b_gray'] * fx_0, 5)}")
    print(f"from cam3 to cam2(color) baseline: {calib['b_rgb']} meters, bf: {round(calib['b_rgb'] * fx_2, 5)}")
