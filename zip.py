import shutil
import os

fontDirectories = os.path.join(os.path.dirname(__file__), "./fonts")
shutil.make_archive("fonts", 'zip', fontDirectories)