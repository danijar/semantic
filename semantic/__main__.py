import argparse
import importlib


def main():
    commands = ['corpus', 'train', 'test', 'visualize', 'recommend']
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=commands)
    args = parser.parse_args()
    command = 'semantic.command.{}.__main__'.format(args.command)
    importlib.import_module(command).main()


if __name__ == '__main__':
    main()
