import nox


@nox.session
def lint(session):
    session.install("-r", "requirements.txt")
    session.run("flake8", "--exclude=env,__pycache__,.nox", "--ignore=E501,W503,E402")
    session.run("pylint", ".", "--disable=invalid-name")


@nox.session
def test(session):
    session.install("-r", "requirements.txt")
    session.run("pytest", "-v", *session.posargs)
