git tag -a v$(python3 setup.py --version) -m "Updated the BGD optimizer to a more general SGD, added batch_size functionality and logistic regression"
git push origin v$(python3 setup.py --version)
