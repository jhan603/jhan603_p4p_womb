# p4_seg

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
