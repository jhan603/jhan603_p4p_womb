# ENGSCI 700A/B Project 41: Make Womb

## Installation

1. Have Anaconda navigator
2. Go to the directory to create this virtual environment in
3. Create a virtual environment by installing python 3.10
   ```python
   conda create -n p4penv python=3.10
   # Then activate the environment
   conda activate p4penv
   ```
4. Install the required dependencies using the requirements.txt file
   ```python
   pip install -r requirements.txt
   ```

## Order of files to be run
1. patch_maker.py
2. train.py
3. Arteries_Edge_Detection.py

## Possible setup issue

### Issue
```bash
ImportError: Failed to import pydot. You must install pydot and graphviz for `pydotprint` to work
```
### Fix:
```bash
pip install pydot pydotplus
winget install graphviz # Only on windows
```
Then:
- Download and install graphviz binaries from [here](https://graphviz.gitlab.io/_pages/Download/Download_windows.html).
- Add path to graphviz bin folder in system PATH

## Comments
1. Use Python conventions when naming variables, methods etc.
	- Variables: Make it meaningful, short and readable. Common short form words like "dir" "img" are acceptable, but otherwise use full form of the word. Snake case, fully lower case (ex: image_paths)
	- Methods: Starts with a verb. Snake case, fully lower case (ex: resize_images())
	- Classes: Try to make it one word if possible. Pascal case (ex: UnetPlusPlus())
2. Try to encapsulate code snippets that repeatedly appear, into methods.
3. Make the methods more granular, dedicating to a single task, where possible.
4. Avoid using hardcoded values, especially in conditions. If you want to change something in the future, it will be hard if that is re-occuring.
5. Don't make paths by using string concatenation because it is depending on the OS. Code will not work if you use in Ubuntu and Windows both. (Not a best practice.) Use os library.
