import os
import shutil
import argparse
import run_tests as tests
import sys
from twine.commands import upload
import json


def copy_wheel(release_name):
    with open("version.json", "r") as f:
        version = json.load(f)

    version_name = '{major}.{build}.{minor}'.format(**version)


    if release_name and release_name.lower() in ['master', 'test', 'development']:
        raise ValueError('May not rename wheel to {}. To deploy a wheel by that name, checkout that branch')

    if not release_name:
        release_name = version_name

    release_name = release_name.replace("-", "_")
    local_file = f"dist/natural_selection-{version_name}.tar.gz"
    remote_file = f"natural_selection-{release_name}.tar.gz"

    os.mkdir('deploy')
    shutil.copy(local_file, f"deploy/{remote_file}")

    print(f"Deploying [{local_file}] -> [{remote_file}]")

    upload.main(['deploy/*', '-u', os.getenv('TWINE_USERNAME'), '-p', os.getenv('TWINE_PASSWORD')])

    version['minor'] += 1
    with open("version.json", "w") as f:
        json.dump(version, f)



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
        os.system("RMDIR deploy /s /q")
    else:
        os.system("rm -rf dist")
        os.system("rm -rf deploy" )

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
        help="Deploys the built package to PyPi",
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
