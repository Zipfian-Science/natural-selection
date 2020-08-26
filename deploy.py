import os
import argparse
from subprocess import check_output
import run_tests as tests
import sys

def copy_wheel(release_name):
    if release_name and release_name.lower() in ['master', 'test', 'development']:
        raise ValueError('May not rename wheel to {}. To deploy a wheel by that name, checkout that branch')
    local_name = check_output(["git", "symbolic-ref", "--short", "HEAD"]).decode("utf8")[0:-1]
    if not release_name:
        release_name = local_name
    release_name = release_name.replace("-", "_")
    local_file = "dist/natural_selection-{release_name}-py3-none-any.whl".format(
        release_name=local_name
    )
    remote_file = "natural_selection-{release_name}-py3-none-any.whl".format(
        release_name=release_name
    )

    print("Deploying [{local_file}] -> [{remote_file}]".format(local_file=local_file, remote_file=remote_file))

    # TODO move somewhere


def lock_and_gen_pipreq():
    os.system("pipenv lock -r > requirements.txt")


def build_wheel():
    delete_build()
    delete_dist()
    os.system("python setup.py sdist bdist_wheel")

def delete_build():
    if sys.platform == "win32":
        os.system("RMDIR build /s /q")
        os.system("RMDIR natural_selection.egg-info /s /q")
    else:
        os.system("rm -rf build")
        os.system("rm -rf natural_selection.egg-info" )

def delete_dist():
    if sys.platform == "win32":
        os.system("RMDIR -f dist /s /q")
    else:
        os.system("rm -rf dist")


def do_unit_tests(args):
    if args.unit:
        return tests.all_unit_tests(args.coverage)
    return True


def do_integration_tests(args):
    if args.integration:
        return tests.all_unit_tests(args.coverage)
    return True


def main(args):
    if not do_unit_tests(args):
        print("Unit testing failed, not building!")
        return

    if not do_integration_tests(args):
        print("Integration testing failed, not building!")
        return

    if args.pipreq:
        lock_and_gen_pipreq()

    build_wheel()

    if args.deploy:
        copy_wheel(args.remotename)

    if args.remove:
        delete_build()
        delete_dist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
        Builds Natural Selection libs package, runs tests
        """,
    )
    parser.add_argument(
        "--remove",
        "-r",
        help="Removes build artifacts",
        action="store_true",
    )
    parser.add_argument(
        "--unit",
        "-u",
        help="Pre-run all unit tests",
        action="store_true",
    )
    parser.add_argument(
        "--integration",
        "-i",
        help="Pre-run all integration tests",
        action="store_true",
    )
    parser.add_argument(
        "--coverage",
        "-c",
        help="Run coverage over tests",
        action="store_true",
    )
    parser.add_argument(
        "--deploy",
        "-d",
        help="Deploys the built package to ***",
        action="store_true",
    )
    parser.add_argument(
        "--remotename",
        type=str,
        default=None,
        help="Define release name for remote object")
    parser.add_argument(
        "--pipreq",
        "-p",
        help="Lock pipenv dependencies and generate requirements.txt",
        action="store_true",
    )
    main(parser.parse_args())
