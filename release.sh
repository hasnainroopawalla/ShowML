git tag -a v$(python3 setup.py --version) -m "Updated the README"
git push origin v$(python3 setup.py --version)
