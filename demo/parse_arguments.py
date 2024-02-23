import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script")

    parser.add_argument("-cc", "--cam_calib", type=str, default='calib_cam0.pkl', help="Path to pickle camera calibration file")
    parser.add_argument("-tp", "--ted_parameters_path", type=str, default='demo_weights/weights_ted.pth.tar', help="Path to ted model weights")
    parser.add_argument("-mp", "--maml_parameters_path", type=str, default='demo_weights/weights_maml', help="Path to maml model weights")
    parser.add_argument("-k", type=int, default=9, help="Number of training points")
    parser.add_argument("-cs", "--camera_size", type=int, nargs=2, default=[640, 480], help="Camera width and height separated by comma")
    parser.add_argument("-dp", "--data_path", type=str, default=None, help="Path to data if collect_data")
    parser.add_argument("-ftp", "--fine_tuning_path", type=str, default=None, help="Person-specific pretrained weights")
    parser.add_argument("-m", "--mode", type=str, choices=["click", "point", "train"], default="point", help="run mode")
    parser.add_argument("-s", "--subject", type=str, default=None, help="Name of subject")
    parser.add_argument("-p", "--num_points", type=int, nargs=2, default=[9, 4], help="Two numbers of calibration points separated by comma")

    args = parser.parse_args()

    print("Parsed arguments:")
    print("cam_calib:", args.cam_calib)
    print("ted_parameters_path:", args.ted_parameters_path)
    print("maml_parameters_path:", args.maml_parameters_path)
    print("k:", args.k)
    width, height = args.camera_size
    print("camera_width:", width)
    print("camera_height:", height)
    
    if args.data_path:
        print("data_path:", args.data_path)
    else:
        print("collect_data:", True)
        print("calibration_points:", *args.num_points)
    if args.fine_tuning_path:
        print("fine_tuning_path:", args.fine_tuning_path)
    else:
        print("fine_tuning:", True)
    print("mode:", args.mode)
    if args.subject:
        print(f'if {args.subject} is ready, we are to start...')
    return args

if __name__ == "__main__":
    parse_arguments()
