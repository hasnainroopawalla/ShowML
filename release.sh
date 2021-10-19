git tag -a v$(python3 setup.py --version) -m "This is a simple module"
git push origin v$(python3 setup.py --version)
