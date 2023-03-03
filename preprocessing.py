from argparse import ArgumentParser
from preprocessing.create_wlasl_landmarks_dataset import parse_create_args, create
from preprocessing.extract_mediapipe_landmarks import parse_extract_args, extract


if __name__ == '__main__':
    main_parser = ArgumentParser()
    subparser = main_parser.add_subparsers(dest="action")
    create_subparser = subparser.add_parser("create")
    extract_subparser = subparser.add_parser("extract")
    parse_create_args(create_subparser)
    parse_extract_args(extract_subparser)

    args = main_parser.parse_args()

    if args.action == "create":
        create(args)
    elif args.action == "extract":
        extract(args)
    else:
        ValueError("action command must be either 'create' or 'extract'")
