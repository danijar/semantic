import argparse
from semantic import command


def main():
    commands = {
        'corpus': command.corpus,
        'train': command.train,
        'test': command.test,
        'visualize': command.visualize,
        'recommend': command.recommend,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choice=list(commands.keys()))
    args = parser.parse_args()
    import args.command.__main__


if __name__ == '__main__':
    main()
