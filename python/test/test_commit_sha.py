import unittest
import git
import alsvinn
import json

## THIS CAN ONLY BE RUN IF YOU HAVE THE GIT SUBMODULE ENABLED
# Do
#    git submodule update --init
#
# If you downloaded a release version of aslvinn, this test is not relevant for you
class TestCommitSha(unittest.TestCase):

    def test_ru(self):
        name = "commit_sha"
        alsvinn_object = alsvinn.run(dimension=[8, 1, 1],
                                     samples=1, name=name)
        with open(f"alsvinncli_{name}_report.json") as f:
            sha_from_run = json.load(f)['report']['revision']

        repo = git.Repo(search_parent_directories=True)
        sha_from_repo = repo.head.object.hexsha
        self.assertEqual(sha_from_run, sha_from_repo)

if __name__ == '__main__':
    unittest.main()
