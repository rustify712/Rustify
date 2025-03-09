import hashlib
import os
import pickle
import time
import zlib

from core.logger.runtime import get_logger

logger = get_logger(name="VCS", filename="vcs.log")

VCS_DIR = ".vcs"


def hash_object(data: bytes) -> str:
    """Computes the SHA-1 hash of the given data.
    Args:
        data (bytes): The data to be hashed.
    Returns:
        str: The hexadecimal representation of the SHA-1 hash.
    """
    sha1 = hashlib.sha1()
    sha1.update(data)
    return sha1.hexdigest()


def store_object(data: bytes, objects_dir: str) -> str:
    """Store the given data in the objects directory.
    Args:
        data (bytes): The data to be stored.
        objects_dir (str): The path to the objects directory.
    Returns:
        str: The SHA-1 hash of the data.
    """
    oid = hash_object(data)
    object_dir = os.path.join(objects_dir, oid[:2])
    object_file = os.path.join(object_dir, oid[2:])
    if not os.path.exists(object_dir):
        os.makedirs(object_dir, exist_ok=True)
    with open(object_file, "wb") as f:
        f.write(zlib.compress(data))
    return oid


def load_object(oid: str, objects_dir: str) -> bytes:
    """Load the object with the given OID.
    Args:
        oid (str): The SHA-1 hash of the object.
        objects_dir (str): The path to the objects directory.
    Returns:
        bytes: The object data.
    """
    object_dir = os.path.join(objects_dir, oid[:2])
    object_file = os.path.join(object_dir, oid[2:])
    if not os.path.exists(object_file):
        raise FileNotFoundError(f"oid({oid}) file {object_file} does not exist")
    with open(object_file, "rb") as f:
        return zlib.decompress(f.read())


class VCS:
    """A primary version control system.

    Args:
        root_dir (str): The path to the root directory of the repository.
    """

    def __init__(self, root_dir: str):
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"folder {root_dir} does not exist")
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"{root_dir} is not a directory")

        self.root_dir = root_dir
        self.vcs_dir = os.path.join(root_dir, VCS_DIR)
        self.objects_dir = os.path.join(self.vcs_dir, "objects")
        self.index_file = os.path.join(self.vcs_dir, "index")
        self.head_file = os.path.join(self.vcs_dir, "HEAD")
        self.log_file = os.path.join(self.vcs_dir, "log")

    def init(self):
        """Initialize the repository."""
        if not os.path.exists(self.vcs_dir):
            # os.makedirs(self.vcs_dir, exist_ok=True)
            os.makedirs(self.objects_dir, exist_ok=True)
            with open(self.index_file, "wb") as f:
                pickle.dump({}, f)
            with open(self.head_file, "w") as f:
                f.write("")
            with open(self.log_file, "w") as f:
                f.write("")
            logger.info(f"initialize the repository {self.root_dir} successfully")
        else:
            logger.warning(f"the repository {self.root_dir} already exists")

    def add(self, filepaths: list[str]):
        """Add files to the index.
        Args:
            filepaths (list[str]): A list of paths to the files to be added.

        Warnings: only support adding files, not directories.
        """
        if not os.path.exists(self.vcs_dir):
            raise FileNotFoundError(f"the repository {self.root_dir} does not exist")

        # load index
        with open(self.index_file, "rb") as f:
            index = pickle.load(f)

        for filepath in filepaths:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"file {filepath} does not exist")
            if os.path.isdir(filepath):
                raise IsADirectoryError(f"{filepath} is a directory")
            with open(filepath, "rb") as f:
                data = f.read()
            oid = store_object(data, self.objects_dir)
            index[filepath] = oid
            logger.info(f"add file {filepath} successfully")

        # save index
        with open(self.index_file, "wb") as f:
            pickle.dump(index, f)

    def commit(self, message):
        """Commit the changes in the index.
        Args:
            message (str): The commit message.
        """
        if not os.path.exists(self.vcs_dir):
            raise FileNotFoundError(f"the repository {self.root_dir} does not exist")
        # load index and create tree object
        with open(self.index_file, "rb") as f:
            index = pickle.load(f)
        tree_data = pickle.dumps(index)
        tree_oid = store_object(tree_data, self.objects_dir)
        # create commit object
        parent = ""
        if os.path.exists(self.head_file):
            with open(self.head_file, "r") as f:
                parent = f.read()

        commit_data = {
            "tree": tree_oid,
            "parent": parent,
            "message": message,
            "timestamp": time.time(),
        }
        commit_data_serialized = pickle.dumps(commit_data)
        commit_oid = store_object(commit_data_serialized, self.objects_dir)
        # update HEAD
        with open(self.head_file, "w") as f:
            f.write(commit_oid)
        # update log
        with open(self.log_file, "a") as f:
            f.write(f"{commit_oid} {message}\n")
        logger.info(f"commit successfully, commit ID: {commit_oid}, message: {message}")

    def log(self):
        """show commit log"""
        if not os.path.exists(self.vcs_dir):
            raise FileNotFoundError(f"the repository {self.root_dir} does not exist")
        with open(self.log_file, "r") as f:
            logs = f.readlines()
        for log in reversed(logs):
            oid, message = log.strip().split(" ", 1)
            print(f"Commit: {oid}\nMessage: {message}\n")

    def checkout(self, oid):
        """Checkout the commit with the given OID.

        Args:
            oid (str): The OID of the commit to be checked out.
        """
        if not os.path.exists(self.vcs_dir):
            raise FileNotFoundError(f"the repository {self.root_dir} does not exist")
        commit_data_serialized = load_object(oid, self.objects_dir)
        commit_data = pickle.loads(commit_data_serialized)
        # load index
        tree_data = load_object(commit_data["tree"], self.objects_dir)
        index = pickle.loads(tree_data)
        # checkout files
        for filepath, fileoid in index.items():
            data = load_object(fileoid, self.objects_dir)
            with open(filepath, "wb") as f:
                f.write(data)
            logger.debug(f"checkout file {filepath} successfully")
        # update HEAD
        with open(self.head_file, "w") as f:
            f.write(oid)
        logger.info(f"checkout commit {oid} successfully")

    def rollback(self):
        """rollback to the previous commit"""
        if not os.path.exists(self.vcs_dir):
            raise FileNotFoundError(f"the repository {self.root_dir} does not exist")
        # load HEAD
        with open(self.head_file, "r") as f:
            current_oid = f.read()
        if not current_oid:
            logger.warning("rollback: no commit yet")
            return
        # load current commit
        current_commit_data_serialized = load_object(current_oid, self.objects_dir)
        current_commit_data = pickle.loads(current_commit_data_serialized)
        parent_oid = current_commit_data["parent"]
        if not parent_oid:
            logger.warning("rollback: no parent commit")
            return
        # checkout parent commit
        self.checkout(parent_oid)
        # update logs and remove the last commit log
        with open(self.log_file, "r") as f:
            logs = f.readlines()
        with open(self.log_file, "w") as f:
            f.writelines(logs[:-1])
        logger.info(f"rollback to commit {parent_oid} successfully")

    def revert(self, filepaths: list[str]):
        """Revert the changes in the given files.
        Args:
            filepaths (list[str]): A list of paths to the files to be reverted.
        """
        if not os.path.exists(self.vcs_dir):
            raise FileNotFoundError(f"the repository {self.root_dir} does not exist")
        # load HEAD
        with open(self.head_file, "r") as f:
            current_oid = f.read()
        if not current_oid:
            logger.warning("revert: no commit yet")
            return
        # load current commit
        current_commit_data_serialized = load_object(current_oid, self.objects_dir)
        current_commit_data = pickle.loads(current_commit_data_serialized)
        # load index
        tree_data = load_object(current_commit_data["tree"], self.objects_dir)
        index = pickle.loads(tree_data)
        # revert files
        for filepath in filepaths:
            if filepath not in index:
                logger.debug(f"revert: file {filepath} not in the current commit")
                continue
            fileoid = index[filepath]
            data = load_object(fileoid, self.objects_dir)
            with open(filepath, "wb") as f:
                f.write(data)
            logger.info(f"revert file {filepath} successfully")

    def status(self):
        """Show the status of the repository."""
        if not os.path.exists(self.vcs_dir):
            raise FileNotFoundError(f"the repository {self.root_dir} does not exist")
        # load index
        with open(self.index_file, "rb") as f:
            index = pickle.load(f)
        changed_files = []
        for filepath, fileoid in index.items():
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    data = f.read()
                current_fileoid = hash_object(data)
                if current_fileoid != fileoid:
                    changed_files.append(filepath)
        if changed_files:
            print("changes to be committed:")
            for filepath in changed_files:
                print(filepath)
        else:
            print("nothing to commit, working directory clean")
