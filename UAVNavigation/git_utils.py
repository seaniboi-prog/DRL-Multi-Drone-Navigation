from git import Repo
import os

PATH_OF_GIT_REPO_PC = r'C:\\Users\\seanf\\Documents\\Workspace\\Python Scripts\\.git'  # make sure .git folder is properly configured
PATH_OF_GIT_REPO_LAPTOP = r'C:\\Users\\seanf\\OneDrive\\Desktop\\School\\Python-Scripts\\.git'  # make sure .git folder is properly configured

def git_pull() -> None:
    try:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_PC))
    except:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_LAPTOP))

    try:
        origin = repo.remote(name='origin')
        origin.pull()
    except:
        print('Some error occurred while pulling the code')

def git_push(commit_message, pull) -> None:
    try:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_PC))
    except:
        repo = Repo(os.path.abspath(PATH_OF_GIT_REPO_LAPTOP))

    try:
        if pull:
            git_pull()

        repo.git.add(all=True)
        repo.index.commit(commit_message)
        origin = repo.remote(name='origin')
        origin.push()
    except:
        print('Some error occurred while pushing the code')

# git_push("Automated Commit Message")
# git_pull()